# QuantTrader-TFT

**QuantTrader-TFT** is a project focused on predicting the prices of major stock market indices, such as the S&P 500, Nasdaq, IBEX 35, Dow Jones, and EURO STOXX 50, using Temporal Fusion Transformer (TFT) models. The goal is to create an investment portfolio that maximizes returns while minimizing risk by leveraging the predictions made by the TFT models.

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

2. **Error**: `× Encountered error while generating package metadata.`
   - **Cause**: This can occur when installing `pytorch-forecasting`.
   - **Solution**: Ensure that your Python version is **3.10.x** or lower.

3. **Error**: `ModuleNotFoundError: No module named 'numpy.lib.function_base'`
   - **Solution**: Update NumPy to the latest version:
     ```bash
     pip install --upgrade numpy
     ```

4. **Error**: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.0 as it may crash."
   - **Solution**: Downgrade NumPy to a compatible version:
     ```bash
     pip install numpy==1.24.3
     ```

5. **Error**: `ModuleNotFoundError: Optuna's integration modules for third-party libraries have started migrating from Optuna...`
   - **Solution**: Install `optuna-integration`:
     ```bash
     pip install optuna-integration
     ```

6. **Error**: `TypeError: Cannot create a consistent method resolution order (MRO) for bases Callback, PyTorchLightningPruningCallback`
   - **Solution**: Install specific versions of the required libraries:
     ```bash
     pip install torch==2.0.1 pytorch-lightning==2.0.2 pytorch_forecasting==1.0.0 torchaudio==2.0.2 torchdata==0.6.1 torchtext==0.15.2 torchvision==0.15.2 optuna==3.4
     ```

