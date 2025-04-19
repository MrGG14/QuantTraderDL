#region imports
from AlgorithmImports import *
#endregion
# Copyright 2019, Hudson and Thames Quantitative Research
# All rights reserved
# Read more: https://github.com/hudson-and-thames/mlfinlab/blob/master/LICENSE.txt
# mlfinlab provided this code file. QuantConnect made small edits to remove some imports.

"""
Implementation of Trend-Scanning labels described in `Advances in Financial Machine Learning: Lecture 3/10
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_
"""
# pylint: disable=invalid-name, too-many-locals

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

#region imports
from AlgorithmImports import *

import plotly.graph_objects as go
#endregion


def rough_daily_backtest(qb, portfolio_weights):
    history = qb.history(
        list(portfolio_weights.columns), 
        portfolio_weights.index[0]-timedelta(1), portfolio_weights.index[-1], 
        Resolution.DAILY
    )['close'].unstack(0)
    # Calculate the daily returns of each asset.
    # `shift` so that it's the return from today to tomorrow.
    asset_daily_returns = history.pct_change(1).shift(-1).dropna() 
    asset_equity_curves = (asset_daily_returns + 1).cumprod() - 1
    strategy_daily_returns = (
        asset_daily_returns * portfolio_weights
    ).sum(axis=1)
    strategy_equity_curve = (strategy_daily_returns + 1).cumprod()
    # Plot the results.
    go.Figure(
        [
            go.Scatter(
                x=asset_equity_curves.index, y=asset_equity_curves[symbol] + 1,
                name=f"Buy-and-hold {str(symbol).split(' ')[0]}"
            )
            for symbol in asset_equity_curves.columns
        ] + [
            go.Scatter(
                x=strategy_equity_curve.index, y=strategy_equity_curve,
                name='Strategy'
            )
        ],
        dict(
            title="Rough Backtest Results<br><sup>The equity curves of "
                + "buy-and-hold for each asset and for the portfolio given "
                + "the asset weights</sup>",
            xaxis_title="Date", yaxis_title="Equity"
        )
    ).show()

def _get_regression_quality_metric(X_subset: np.array, y_subset: np.array, metric: str) -> tuple:
    # pylint: disable=invalid-name
    """
    Get goodness of fit metric for a linear regression fit on `X_subset`, `y_subset`.

    :param X_subset: (np.array) Array of X values (1,2,3,...) for linear regression.
    :param y_subset: (np.array) Array of y values for linear regression.
    :param metric: (str) Metric used to define top-fit regression.
    :return: (tuple) Goodness of fit metric and regression slope t-value.
    """

    # Get regression coefficients estimates
    res = sm.OLS(y_subset, X_subset).fit()
    # Check if l gives the maximum t-value among all values {0...L}
    t_beta_1 = res.tvalues[1]
    if metric == 't_value':
        compare_value = abs(t_beta_1)
    # TODO: Refactor to not have duplicate lines fitting linear regression,
    # but without compromising performance for `metric = 't_value'`.
    elif metric == 'mean_absolute_error':
        # We maximize -MAE/MSE
        linear_model = LinearRegression()
        linear_model.fit(X_subset, y_subset)
        compare_value = -1 * mean_absolute_error(y_subset, linear_model.predict(X_subset))
    elif metric == 'mean_squared_error':
        # We maximize -MAE/MSE
        linear_model = LinearRegression()
        linear_model.fit(X_subset, y_subset)
        compare_value = -1 * mean_squared_error(y_subset, linear_model.predict(X_subset))
    else:
        raise ValueError(
            "Unknown comparison metric. Possible options: t_value, mean_absolute_error, mean_squared_error")
    return compare_value, t_beta_1

def classify_trend(metric, compare_value, b1, mse_values, 
                   t_threshold=2, mse_percentile=75, b1_threshold=0.001):
    """
    Función para asignar la etiqueta de tendencia basada en la métrica seleccionada.
    Permite parametrizar los umbrales para diferentes métricas.
    """
    if metric == 't_value':
        if abs(compare_value) < t_threshold:  # Si el t-value no es significativo
            return 0
    elif metric in ['mean_squared_error', 'mean_absolute_error']:
        mse_threshold = np.percentile(mse_values, mse_percentile)  # Umbral dinámico basado en percentiles
        if compare_value > mse_threshold:  # Si el error es alto, no hay tendencia clara
            return 0
    
    # Si pasa el umbral de la métrica, usamos b1 para definir la dirección
    if abs(b1) < b1_threshold:
        return 0
    return np.sign(b1)  # 1 si b1>0, -1 si b1<0


def trend_scanning_labels(price_series: pd.Series, t_events: list = None, observation_window: int = 20,
                          metric: str = 't_value', look_forward: bool = True, min_sample_length: int = 5,
                          step: int = 1, t_threshold=2, mse_percentile=75, b1_threshold=0.001) -> pd.DataFrame:
    """
    Implementa Trend Scanning Labels con diferentes métricas y umbral generalizado.
    Ahora permite parametrizar los thresholds para mejor ajuste.
    """
    if t_events is None:
        t_events = price_series.index

    t1_array, t_values_array, b1_array, mse_values = [], [], [], []

    for obs_id, index in enumerate(t_events):
        if look_forward:
            subset = price_series.loc[index:].iloc[:observation_window]
        else:
            subset = price_series.loc[:index].iloc[max(0, obs_id - observation_window):]

        if subset.shape[0] >= min_sample_length:
            max_compare_value = -np.inf if metric == 't_value' else np.inf
            best_window, best_b1 = None, None

            for forward_window in np.arange(min_sample_length, subset.shape[0], step):
                y_subset = subset.iloc[:forward_window].values.reshape(-1, 1) if look_forward else subset.iloc[-forward_window:].values.reshape(-1, 1)
                X_subset = np.ones((y_subset.shape[0], 2))
                X_subset[:, 1] = np.arange(y_subset.shape[0])

                compare_value, b1 = _get_regression_quality_metric(X_subset, y_subset, metric)
                mse_values.append(compare_value)

                if ((metric == 't_value' and compare_value > max_compare_value) or
                    (metric in ['mean_squared_error', 'mean_absolute_error'] and compare_value < max_compare_value)):
                    max_compare_value, best_window, best_b1 = compare_value, forward_window, b1

            if best_window is not None:
                label_endtime_index = subset.index[best_window - 1]
                t1_array.append(label_endtime_index)
                t_values_array.append(max_compare_value)
                b1_array.append(best_b1)
            else:
                t1_array.append(None)
                t_values_array.append(None)
                b1_array.append(None)

        else:
            t1_array.append(None)
            t_values_array.append(None)
            b1_array.append(None)

    labels = pd.DataFrame({'t1': t1_array, 'metric_value': t_values_array, 'b1': b1_array}, index=t_events)

    labels['bin'] = labels.apply(lambda row: classify_trend(metric, row['metric_value'], row['b1'], mse_values,
                                                            t_threshold, mse_percentile, b1_threshold), axis=1)

    # Debug: Verificar cuántos labels 0 hay
    print(f"Tendencias Neutras (bin=0): {sum(labels['bin'] == 0)} / {len(labels)} ({round(sum(labels['bin'] == 0) / len(labels) * 100, 2)}%)")

    return labels
