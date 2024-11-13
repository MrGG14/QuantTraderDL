# TFT-QuantTrader

**TFT-QuantTrader** is a project focused on predicting the prices of major stock market indices, such as the S&P 500, Nasdaq, IBEX 35, Dow Jones, and EURO STOXX 50, using Temporal Fusion Transformer (TFT) models. The goal is to create an investment portfolio that maximizes returns while minimizing risk by leveraging the predictions made by the TFT models.

The models provide predictions with different confidence intervals, which will be used to identify indices with upward trends that are more likely to be profitable. Future enhancements will include strategies to detect significant trend changes.

## Key Features

- **Time Series Forecasting**: Using Temporal Fusion Transformers (TFT) to forecast future prices of indices like S&P 500, Nasdaq, IBEX 35, and others.
  
- **Portfolio Optimization**: Build a portfolio that maximizes gains and minimizes risks based on predicted prices and confidence intervals.

- **Confidence Interval-based Decision Making**: Focus on indices with the most promising upward trends based on prediction confidence intervals.

- **Risk Minimization**: Manage risk by adjusting exposure to indices with high uncertainty or low-confidence predictions.

- **Future Strategy Enhancements**: Plans to implement strategies for detecting significant trend reversals or sudden changes in market behavior.

## Project Structure

- **Data**: Code for collecting historical data for the stock market indices and preparing it for training.
  
- **Model**: Implementation of the Temporal Fusion Transformer (TFT) model for time series forecasting.

- **Prediction**: Scripts to make predictions of future index prices based on trained models.

- **Portfolio Construction**: Code to optimize an investment portfolio using the model's predictions while considering risk and maximizing potential returns.

- **Risk Analysis**: Evaluation of predicted price trends to determine the level of risk associated with different investment choices.

- **Future Work**: Detection of significant trend changes, backtesting portfolio strategies, and enhancing prediction models.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/TFT-QuantTrader.git
    ```
