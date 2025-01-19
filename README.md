# QuantTraderDL: Forecasting and Trading Financial Indices with Advanced AI Models

**QuantTraderDL** is a comprehensive project aimed at leveraging AI for smarter investment strategies in financial markets. The repository combines two cutting-edge approaches:

1. **Temporal Fusion Transformer (TFT)**: A model for forecasting the prices of major stock market indices (e.g., S&P 500, Nasdaq, IBEX 35).
2. **Deep Reinforcement Learning (DRL) Trading Bot**: A trading bot that learns to make intelligent decisions based on market dynamics.

Although the forecasting and trading components are independent, future iterations aim to integrate them into a unified framework.



## Table of Contents

1. [Project Overview](#project-overview)
2. [Temporal Fusion Transformer for Financial Forecasting](#temporal-fusion-transformer-for-financial-forecasting)
3. [Reinforcement Learning Trading Bot](#reinforcement-learning-trading-bot)
4. [Variables Used for Model Training](#variables-used-for-model-training)
5. [Key Features](#key-features)
6. [Project Structure](#project-structure)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Installation](#installation)
10. [Common Import Errors and Solutions](#common-import-errors-and-solutions)



## Project Overview

**QuantTrader-TFT** aims to tackle the challenges of financial forecasting and trading using state-of-the-art AI techniques. The project consists of two primary components:

1. **TFT Forecasting**: Predicts multi-horizon trends in major financial indices, providing insights into market movements and enabling data-driven portfolio construction.
2. **DRL Trading Bot**: Learns optimal trading strategies through deep reinforcement learning, enabling dynamic responses to market conditions.



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

1. **Integration of TFT and DRL**: Combining forecasting insights with trading strategies.
2. **Real-Time Capabilities**: Adapting both models for live market data.
3. **Advanced Features**: Incorporating macroeconomic indicators and trend reversal detection.



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

1. **Error**: `Cannot import name '_centered' from 'scipy.signal.signaltools'`
   - **Solution**: Add the `_centered` function manually in `scipy.signal.signaltools`.

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

---

With these capabilities, **QuantTrader-TFT** aspires to provide a robust foundation for AI-driven financial forecasting and trading strategies.
