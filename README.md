# QuantTraderDL: Quantitative and AI-Driven Forecasting and Trading of Financial Indices

**QuantTraderDL** is an educational and research-oriented project combining quantitative finance theory and state-of-the-art AI to develop intelligent financial strategies. It offers a practical introduction to both traditional quantitative methods and deep learning models for forecasting and trading:

1. **QuantConnect Strategies**: A collection of algorithmic trading strategies implemented using the QuantConnect platform, serving as a foundational exploration into quantitative trading methodologies.
2. **Temporal Fusion Transformer (TFT)**: A model for forecasting the prices of major stock market indices (e.g., S&P 500, Nasdaq, IBEX 35).
3. **Deep Reinforcement Learning (DRL) Trading Bot**: A trading bot that learns to make intelligent decisions based on market dynamics.

Although the forecasting and trading components are independent, future iterations aim to integrate them into a unified framework.

## Table of Contents

1. [Project Overview](#project-overview)
2. [QuantConnect Strategies](#quantconnect-strategies)
4. [Temporal Fusion Transformer for Financial Forecasting](#temporal-fusion-transformer-for-financial-forecasting)
5. [Reinforcement Learning Trading Bot](#reinforcement-learning-trading-bot)
6. [Variables Used for Model Training](#variables-used-for-model-training)
7. [Key Features](#key-features)
8. [Project Structure](#project-structure)
9. [Results](#results)
10. [Future Work](#future-work)
11. [Installation](#installation)
12. [Common Import Errors and Solutions](#common-import-errors-and-solutions)

## Project Overview

**QuantTrader-TFT** addresses the challenges of quantitative investment strategy development by integrating traditional finance methodologies with state-of-the-art AI techniques. The project consists of three primary components:

1. **QuantConnect Strategies**: A growing repository of algorithmic trading strategies implemented on QuantConnect. These are designed to explore various foundational and advanced concepts in quantitative finance — from momentum and mean reversion to factor models and portfolio optimization. While these strategies are kept isolated for educational clarity, real-world systems often combine multiple strategies into a cohesive, risk-aware trading engine.
2.  **TFT Forecasting**: Predicts multi-horizon trends in major financial indices, providing insights into market movements and enabling data-driven portfolio construction.
3. **DRL Trading Bot**: Learns optimal trading strategies through deep reinforcement learning, enabling dynamic responses to market conditions.

## QuantConnect Strategies

The `QuantConnectStrategies/` directory contains a diverse set of algorithmic trading strategies built for the QuantConnect platform. These are used to study the logic and behavior of quantitative strategies in isolation.

### 🔍 Strategy Categories and Examples

The `QuantConnectStrategies/` directory contains a curated set of algorithmic trading experiments designed to explore diverse quantitative methodologies using the QuantConnect platform. Each notebook reflects a unique research angle — from econometric modeling to deep learning and NLP applications.

These strategies are **meant for educational and experimental purposes**, often studied in isolation. In real-world trading systems, the most robust portfolios integrate multiple strategies, risk models, and execution logic.

To facilitate exploration and understanding, the various QuantConnect trading strategies are organized into categories based on their core approach and methodology. This categorization helps to quickly identify strategies that share similar themes, such as momentum, risk management, or machine learning techniques. Each category groups strategies that target specific market behaviors or leverage particular quantitative methods.


| Category                         | Strategy Names                                                                                                                        | Description                                                                                             |
|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| **Trend Following / Momentum**   | `01_Trend_Scanning`, `03_Reversion_vs_Trending_Strategy_Selection`, `14_Temporal_CNN_Prediction`                                      | Exploit market trend continuations by positioning in the direction of momentum.                          |
| **Regime Detection / Market States** | `02_Factor_Preprocessing_Techniques_for_Regime_Detection`, `04_Hidden_Markov_Models`                                                  | Identify changing market regimes or conditions to adapt strategies accordingly.                         |
| **Mean Reversion / Statistical Arbitrage** | `03_Reversion_vs_Trending_Strategy_Selection`, `13_PCA_Statistical_Arbitrage_Mean_Reversion`                                          | Seek profits by exploiting price reversions to mean or statistical spreads.                            |
| **Machine Learning & Classification** | `05_FX_SVM_Wavelet_Forecasting`, `09_ML_Trading_Pairs_Selection`, `15_Gaussian_Classifier_for_Direction_Prediction`, `10_Stock_Selection_through_Clustering_Fundamental_Data` | Use ML techniques for forecasting and asset/pairs selection.                                          |
| **Technical Pattern Recognition** | `17_Head_Shoulders_Pattern_Matching_with_CNN`                                                                                        | Detect classic technical chart patterns to generate trading signals.                                  |
| **Dividend / Yield Strategies**  | `06_Dividend_Harvesting_Selection_of_High-Yield_Assets`                                                                              | Focus on selecting high dividend yield assets, especially around ex-dividend dates.                    |
| **Risk Management / Volatility** | `07_Effect_of_Positive-Negative_Splits`, `08_Stoploss_Based_on_Historical_Volatility_and_Drawdown_Recovery`, `11_Inverse_Volatility_Rank_and_Allocate_to_Future_Contracts` | Manage risk by adjusting exposure based on volatility, drawdown recovery, and asymmetric signal effects.|
| **Transaction Costs / Execution** | `12_Trading_Costs_Optimization`                                                                                                      | Simulate and optimize slippage and transaction costs to improve portfolio rebalancing.                 |
| **Natural Language Processing (NLP)** | `16_LLM_Summarization_of_Tiingo_News_Articles`, `19_FinBERT_Model`                                                                    | Use language models to analyze and summarize financial news for sentiment and event-driven signals.    |
| **Advanced Time Series Models**  | `18_Amazon_Chronos_Model`                                                                                                             | Experiment with Amazon’s Chronos model for advanced market time series forecasting.                    |

💡 These strategies serve as a lab for testing theoretical ideas in practice. They prioritize interpretability, modeling variety, and research reproducibility over short-term performance.

For general guidelines and API documentation, visit the [QuantConnect Strategy Guide](https://www.quantconnect.com/docs/v2/writing-algorithms/introduction).

These scripts are meant as blueprints for further development. A successful quant system often blends multiple strategies, adds robust risk management, and adapts to changing market conditions.

## Temporal Fusion Transformer for Financial Forecasting

The **Temporal Fusion Transformer (TFT)** is a deep learning model designed for multi-horizon time series forecasting. Its advanced architecture combines attention mechanisms with recurrent layers, capturing short- and long-term dependencies to provide accurate forecasts with confidence intervals.

![tft](https://github.com/user-attachments/assets/9518091a-f3e3-4eb3-83a0-fe3e087d5edb)

### Advantages of TFT

- **Multi-Horizon Forecasting**: Predicts trends over multiple time steps, supporting long-term planning.
- **Uncertainty Quantification**: Confidence intervals allow for risk-aware decision-making.
- **Feature Integration**: Leverages static, time-varying known, and time-varying unknown variables to enhance prediction accuracy.

### Application to Financial Indices

TFT models in this project forecast trends for indices like the S&P 500, Nasdaq, and IBEX 35, enabling:

- **Maximized Returns**: Identifying high-confidence upward trends.
- **Risk Management**: Avoiding indices with high forecast uncertainty.



## Reinforcement Learning Trading Bot

The **DRL Trading Bot** is designed to make intelligent buy, sell, or hold decisions based on market dynamics. It uses reinforcement learning to maximize cumulative returns by:

- Learning optimal strategies through interaction with historical data.
- Adapting to changing market conditions.

### Features of the Trading Bot

- **Reward Optimization**: Focuses on maximizing returns while minimizing drawdowns.
- **Environment Interaction**: Simulates a trading environment using historical data to train the bot.
- **Scalable Framework**: Supports integration with real-time trading systems in future iterations.



## Variables Used for Model Training

This section describes the variables used to train price prediction models and the trading bot. These variables include technical indicators, macroeconomic data, and derived features that capture complex patterns in financial time series.

### Macroeconomic Variables

- **GDP of influential countries**: Gross Domestic Product (GDP) of several key economies, used to assess the overall state of the global economy and its impact on financial markets.
- **Federal Reserve Interest Rate (FEDFUNDS)**: Reflects U.S. monetary policy and its influence on liquidity and risk appetite.

### Technical Indicators

- **Simple Moving Average (SMA)**:
  - SMA_50 and SMA_200: Price averages calculated over 50- and 200-day windows, respectively. These averages are used to identify long-term trends.
- **Exponential Moving Average (EMA)**: Similar to SMA but gives more weight to recent prices, making it more responsive to changes.
- **Relative Strength Index (RSI)**: An oscillator that measures the speed and magnitude of price movements, helping to identify overbought (>70) or oversold (<30) conditions.
- **Bollinger Bands**:
  - Upper and lower bands reflect price volatility over a specific time window.
- **Moving Average Convergence Divergence (MACD)**: A momentum indicator that combines moving averages to detect bullish and bearish crossovers.
- **ATR (Average True Range)**: Measures recent price volatility, providing key information on market risk.
- **CCI (Commodity Channel Index)**: Identifies overbought or oversold levels relative to a statistical average.
- **ROC (Rate of Change)**: Measures the percentage change in price over a specified period.
- **Williams %R**: A momentum indicator showing overbought and oversold levels relative to high and low prices.
- **Stochastic Oscillator**: Compares the closing price to a range of prices over a period, useful for spotting reversals.

### Derived and Smoothed Variables

- **Gaussian Filters**: Applied to the target price series to smooth out volatility and highlight underlying trends.
  - Gaussian filter formula:

    $$
    \text{G}(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{x^2}{2\sigma^2}}
    $$

    Where \(\sigma\) controls the level of smoothing.
- **Lags of the target variable**: Lagged values of the target price that capture market inertia and repetitive patterns.
- **Smoothed moving averages (target_smoothed)**: Additional smoothing of the target variable to enhance model stability.

### Bullish and Bearish Trends

Signals are generated based on defined rules to identify clear trends:

- **Moving Averages**:
  - Bullish: When SMA_50 > SMA_200.
  - Bearish: When SMA_50 < SMA_200.
- **RSI**:
  - Bullish: RSI < 30 (oversold).
  - Bearish: RSI > 70 (overbought).
- **Bollinger Bands**:
  - Bullish: Price is below the lower band.
  - Bearish: Price is above the upper band.
- **MACD**:
  - Bullish: MACD > Signal Line.
  - Bearish: MACD < Signal Line.
- **ATR**:
  - Bullish: Current ATR > ATR moving average.
  - Bearish: Current ATR < ATR moving average.

### Exogenous Variables

- **External stock indices**: Data from other relevant indices, used as exogenous variables to improve the model's predictive power.

These variables are crucial to capturing the various factors that influence financial market movements, ranging from macroeconomic data to technical signals and derived trends.


## Key Features

1. **Forecasting with TFT**: High-accuracy price predictions for major financial indices.
2. **Trading with DRL**: Intelligent decision-making for stock market trading.
3. **Risk Management**: Confidence intervals in forecasts and adaptive trading strategies.
4. **Future Integration**: Plans to combine TFT predictions with DRL trading for end-to-end automation.



## Project Structure

- **`data/`**: Historical market data preparation for both forecasting and trading.
- **`plots/`**: Visualizations of TFT predictions.
- **`results/`**: Evaluation metrics and hyperparameter configurations for reproducibility.
- **`alternatives/`**: Exploratory models (e.g., LSTM) for comparison.
- **`Stock-Price-Forecasting-TFT/`**: Codebase for training and evaluating TFT models.
- **`RL-Trading-Bot/`**: Implementation of the deep reinforcement learning trading bot.



## Results

### TFT Forecasting Results

- Achieved a test MAE of **53.15** for the S&P 500 index over a 5-month period.
- Example prediction performance:

![Example Prediction](https://github.com/user-attachments/assets/3faf9b0c-b0f8-4411-91d2-c523965146dc)

### DRL Trading Bot Results

- Working on it!



## Future Work

Future plans for QuantTrader-TFT include:

- **Significant Trend Change Detection**: Developing methods to detect sharp trend changes or reversals to adjust the portfolio strategy dynamically.
- **Enhanced Economic Indicators**: Adding new macroeconomic and market indicators as input features for improved accuracy.
- **Backtesting and Real-Time Forecasting**: Implementing backtesting for performance evaluation and adding real-time forecasting for adaptive portfolio management.

--- 

With these capabilities, **QuantTrader-TFT** aims to build a reliable framework for forecasting financial indices and creating optimized, risk-aware investment strategies.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/MrGG14/QuantTrader-TFT.git
    ```

2. Navigate to the directory:

    ```bash
    cd QuantTrader-TFT
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```



## Common Import Errors and Solutions

When running the script, you may encounter some common import errors. Below is a list of known issues and recommended solutions:

1. **Error**: `Cannot import name '_centered' from 'scipy.signal.signaltools'`
   - **Solution**: Manually define and add the `_centered` function in `scipy.signal.signaltools`:
     ```python
     def _centered(arr, newsize):
         # Return the center newsize portion of the array.
         newsize = np.asarray(newsize)
         currsize = np.array(arr.shape)
         startind = (currsize - newsize) // 2
         endind = startind + newsize
         myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
         return arr[tuple(myslice)]

     scipy.signal.signaltools._centered = _centered
     ```

2. **Error**: `Encountered error while generating package metadata.`
   - **Solution**: Use Python **3.10.x** or lower.

3. **Error**: `ModuleNotFoundError: No module named 'numpy.lib.function_base'`
   - **Solution**: Update NumPy:
     ```bash
     pip install --upgrade numpy
     ```

4. **Error**: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.0."
   - **Solution**: Downgrade NumPy:
     ```bash
     pip install numpy==1.24.3
     ```

5. **Error**: `ModuleNotFoundError: Optuna's integration modules for third-party libraries...`
   - **Solution**: Install `optuna-integration`:
     ```bash
     pip install optuna-integration
     ```

6. **Error**: `TypeError: Cannot create a consistent method resolution order (MRO)...`
   - **Solution**: Install specific library versions:
     ```bash
     pip install torch==2.0.1 pytorch-lightning==2.0.2 pytorch_forecasting==1.0.0
     ```

6. **Error**: `ImportError: Missing optional dependency 'xlrd'. Install xlrd >= 2.0.1 for xls Excel support Use pip or conda to install xlrd.`
   - **Solution**: Install specific versions of the required libraries:
     ```bash
     pip install xlrd >= 2.0.1
     ```
With these capabilities, **QuantTrader-TFT** aspires to provide a robust foundation for AI-driven financial forecasting and trading strategies.

