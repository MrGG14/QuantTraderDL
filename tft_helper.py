import scipy.signal.signaltools
import numpy as np
from datetime import timedelta
def _centered(arr, newsize):
    # Return the center newsize portion of the array.
    newsize = np.asarray(newsize)
    currsize = np.array(arr.shape)
    startind = (currsize - newsize) // 2
    endind = startind + newsize
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return arr[tuple(myslice)]


scipy.signal.signaltools._centered = _centered

import pandas as pd
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)
from pytorch_forecasting import TemporalFusionTransformer
import lightning.pytorch as pl
import torch
from pytorch_forecasting.metrics import QuantileLoss, MAE, MAPE
from lightning.pytorch.tuner import Tuner
from scipy.signal.signaltools import _centered
import random
import os
import csv
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import pickle
import warnings

# import pandas_ta as ta

# import talib as tal


def get_best_lr(train, train_dataloader, val_dataloader, max_lr=10.0, min_lr=1e-6, **kwargs):

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        gradient_clip_val=kwargs.get("gradient_clip_val", 0.1),
    )
    tft = TemporalFusionTransformer.from_dataset(
        train,
        learning_rate=kwargs.get("learning_rate", 0.15),
        hidden_size=kwargs.get("hidden_size", 16),
        attention_head_size=kwargs.get("attention_head_size", 2),
        dropout=kwargs.get("dropout", 0.1),
        hidden_continuous_size=kwargs.get("hidden_continuous_size", 8),
        loss=kwargs.get("loss", QuantileLoss()),
        log_interval=kwargs.get("log_interval", 10),
        optimizer=kwargs.get("optimizer", "Ranger"),
        reduce_on_plateau_patience=kwargs.get("reduce_on_plateau_patience", 4),
    )

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=max_lr,
        min_lr=min_lr,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

    return res.suggestion()


def tft_trainer(
    train,
    train_dataloader,
    val_dataloader,
    max_epochs=20,
    model_path=None,
    **kwargs,
):
    
    torch.backends.cudnn.benchmark = True

    # Filtrar los callbacks que no son None
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        # precision=16,
        enable_model_summary=False,
        gradient_clip_val=kwargs.get("gradient_clip_val", 0.1),
        limit_train_batches=32,
        callbacks=[
            callback
            for callback in [
                kwargs.get("lr_logger"),
                kwargs.get("early_stop_callback"),
            ]
            if callback is not None
        ],
    )

    tft = TemporalFusionTransformer.from_dataset(
        train,
        learning_rate=kwargs.get("learning_rate", 0.15),
        hidden_size=kwargs.get("hidden_size", 16),
        attention_head_size=kwargs.get("attention_head_size", 2),
        dropout=kwargs.get("dropout", 0.1),
        hidden_continuous_size=kwargs.get("hidden_continuous_size", 8),
        loss=kwargs.get("loss", kwargs.get("loss", QuantileLoss())),
        log_interval=kwargs.get("log_interval", 64),
        optimizer=kwargs.get("optimizer", "Ranger"),
        reduce_on_plateau_patience=kwargs.get("reduce_on_plateau_patience", 4),
    )

    trainer.fit(tft, train_dataloader, val_dataloader)

    # Guardar el modelo entrenado
    if model_path:
        model_path = model_path
        torch.save(tft.state_dict(), model_path)

    loss = (
        trainer.callback_metrics["val_loss"].item()
        if "train_loss_epoch" in trainer.callback_metrics
        else None
    )

    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    return tft, loss


def tft_predict(tft, data, n_preds=None):
    predictions = tft.predict(data, mode="raw", return_x=True)

    for idx in range(
        n_preds if n_preds is not None else predictions.output[0].shape[0]
    ):  # plot n_preds or  all
        tft.plot_prediction(predictions.x, predictions.output, idx=idx, add_loss_to_title=True)
    return predictions



def save_exp_results(exp_path, tft_params, model_days, n_prev_hours, group, val_loss, epochs):
    tft_exps = pd.read_excel(exp_path)

    model_name = f"model_{group}_{model_days}_{n_prev_hours}"
    loss = val_loss

    new_exp = {"model_name": model_name, "loss": loss, "epochs": epochs}
    new_exp.update(tft_params)
    tft_exps = tft_exps.append(new_exp, ignore_index=True)

    tft_exps.sort_values(by="loss", ascending=True, inplace=True)
    tft_exps = tft_exps.reset_index(drop=True)

    tft_exps.to_excel(exp_path, index=False)


def buildLaggedFeatures(s, lag=2, cols=[], dropna=True):
    """
    Builds a new DataFrame to facilitate regressing over all possible lagged features
    """
    if type(s) is pd.DataFrame:
        new_dict = {}
        for col_name in cols:
            # new_dict[col_name] = s[col_name]
            # create lagged Series
            for l in range(1, lag + 1):
                new_dict["%s_lag%d" % (col_name, l)] = s[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=s.index)

    elif type(s) is pd.Series:
        the_range = range(lag + 1)
        res = pd.concat([s.shift(i) for i in the_range], axis=1)
        res.columns = ["lag_%d" % i for i in the_range]
    else:
        print("Only works for DataFrame or Series")
        return None
    if dropna:
        return res.dropna()
    else:
        return res


def make_preds(
    train,
    test,
    model,
    encoder_lenght,
    test_lenght,
    pred_lenght,
    quantiles: bool = True,
):
    try:
        group = model.output_transformer.groups[0]
    except:
        group = 'month'
    if quantiles:
        try:  # for Quantileloss
            preds = []
            preds_data = pd.concat([train[-encoder_lenght:], test])
            for i in range(0, test_lenght, pred_lenght):
                new_data = preds_data[i : i + encoder_lenght + pred_lenght]
                new_data.loc[:, group] = new_data.iloc[0, new_data.columns.get_loc(group)]
                new_raw_predictions = model.predict(new_data, mode="raw", return_x=True)
                prediction = []
                for i in range(pred_lenght):
                    prediction.append(float(new_raw_predictions.output.prediction[0][i][3]))
                preds.append(prediction)
        except:  # for MQF2DistributionLoss
            preds = []
            preds_data = pd.concat([train[-encoder_lenght:], test])
            for i in range(0, test_lenght, pred_lenght):
                new_data = preds_data[i : i + encoder_lenght + pred_lenght]
                new_data.loc[:, group] = new_data.iloc[0, new_data.columns.get_loc(group)]
                prediction = model.to_prediction(new_raw_predictions.output)[0].flatten().tolist()
                preds.append(prediction)
    else:
        preds = []
        preds_data = pd.concat([train[-encoder_lenght:], test])
        for i in range(0, test_lenght, pred_lenght):
            new_data = preds_data[i : i + encoder_lenght + pred_lenght]
            new_data.loc[:, group] = new_data.iloc[0, new_data.columns.get_loc(group)]
            new_raw_predictions = model.predict(new_data, mode="raw", return_x=True)
            prediction = new_raw_predictions.output.prediction[0].flatten().tolist()
            preds.append(prediction)

    return preds


def random_hyperparameter_search(
    data,
    train,
    train_dataloader,
    val_dataloader,
    test,
    param_grid,
    n_iterations=10,  # Número de iteraciones aleatorias
    max_epochs=50,
    save_dir="plots",
    csv_file="resultados.csv",
):
    best_val_loss = float("inf")
    best_params = None
    best_model = None

    # Crear el directorio para guardar las imágenes si no existe
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Si el archivo CSV no existe, crear uno nuevo con encabezados
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Fecha",
                    "ID",
                    "Gradient Clip Val",
                    "Hidden Size",
                    "Dropout",
                    "Hidden Continuous Size",
                    "Attention Head Size",
                    "Learning Rate",
                    "Loss Function",
                    "Epochs",
                    "Val Loss",
                    "Test MAE",
                    "Test MAPE",
                    "Test RMSE",
                    "Training Time (s)",
                ]
            )
# Cargar combinaciones ya evaluadas en un conjunto
    tried_combinations = set()
    initial_idx = 0  # Para el ID inicial

    with open(csv_file, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Crear un tuple con los valores de los hiperparámetros relevantes
            combination = (
                float(row["Gradient Clip Val"]),
                int(row["Hidden Size"]),
                float(row["Dropout"]),
                int(row["Hidden Continuous Size"]),
                int(row["Attention Head Size"]),
                float(row["Learning Rate"]),
                str(row["Loss Function"]),
            )
            tried_combinations.add(combination)
            initial_idx = max(initial_idx, int(row["ID"]))

    param_keys = list(param_grid.keys())
    param_values = [param_grid[key] for key in param_keys]

    for idx in range(n_iterations):
        # Seleccionar aleatoriamente una combinación de hiperparámetros
        params = {key: random.choice(values) for key, values in zip(param_keys, param_values)}

        # Crear un tuple con la combinación de hiperparámetros actuales
        current_combination = (
            params["gradient_clip_val"],
            params["hidden_size"],
            params["dropout"],
            params["hidden_continuous_size"],
            params["attention_head_size"],
            params["learning_rate"],
            params["loss"].__class__.__name__,
        )

        # Verificar si la combinación ya fue probada
        if current_combination in tried_combinations:
            print(f"Combinación ya probada, saltando: {params}")
            continue  # Saltar esta iteración si la combinación ya existe en el CSV
        
        tried_combinations.add(current_combination)
        print(f"\n -------------------------------------------------------- \n Probando combinación aleatoria {idx+1}/{n_iterations}: {params}")

        # Medir el tiempo de inicio del entrenamiento
        start_time = datetime.now()

        # Entrenar el modelo con la combinación actual de hiperparámetros
        tft, val_loss = tft_trainer(
            train, train_dataloader, val_dataloader, max_epochs=max_epochs, **params
        )

        # Medir el tiempo de finalización del entrenamiento
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        minutes, seconds = divmod(int(training_time), 60)
        print(f'Training time: {minutes}m {seconds}s')       
        
        # Generar predicciones en los datos de test usando los parámetros actuales. 
        try:
            preds = make_preds(
                train=data,
                test=test,
                model=tft,
                encoder_lenght=params["n_prev_len"],
                test_lenght=params["test_len"],
                pred_lenght=params["pred_len"],
                quantiles=True if isinstance(params["loss"], QuantileLoss) else False,
            )

            # Calcular métricas de error en test
            preds_flat = [item for sublist in preds for item in sublist]
            real_vals = test["target"].to_list()
            test_mae = mean_absolute_error(real_vals, preds_flat)
            test_mape = mean_absolute_percentage_error(real_vals, preds_flat)
            test_rmse = mean_squared_error(real_vals, preds_flat, squared=False)

            # Verificar si la pérdida es la mejor hasta ahora
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                best_model = tft
                print(f"Nueva mejor combinación encontrada: {params} con pérdida {val_loss:.4f}")

            # Guardar los resultados de la iteración en el archivo CSV
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        initial_idx + idx + 1,
                        params["gradient_clip_val"],
                        params["hidden_size"],
                        params["dropout"],
                        params["hidden_continuous_size"],
                        params["attention_head_size"],
                        params["learning_rate"],
                        params["loss"].__class__.__name__,
                        max_epochs,
                        val_loss,
                        test_mae,
                        test_mape,
                        test_rmse,
                        training_time,
                    ]
                )

            # Crear y guardar el gráfico de esta iteración
            dates = test["Date"].to_list()
            plt.figure(figsize=(10, 6))
            plt.plot(dates, preds_flat, color="r", label="Predicciones", marker="o", linestyle="--")
            plt.plot(dates, real_vals, color="g", label="Valores Reales", marker="x", linestyle="-")

            num_barras = len(dates) // params["pred_len"]
            for i in range(1, num_barras + 1):
                pos = i * 25
                if pos < len(dates):  # Asegurarse de no exceder el rango
                    plt.axvline(x=dates[pos], color="b", linestyle="--", linewidth=0.8)
                    plt.text(dates[pos] - timedelta(days=1), max(preds_flat)*1.02, f'Pred {i}', rotation=90, ha='left', color="blue", fontsize=8)

            # x_ticks = [dates[i * 25 - 1] for i in range(1, num_barras + 1)]
            # x_labels = [f'Preds month {i}' for i in range(1, num_barras + 1)]
            # plt.xticks(ticks=x_ticks, labels=x_labels, rotation=45, ha='right', fontsize=8)

            # plt.ylim(bottom=0)
            plt.title(f"Predicciones vs Valores Reales - Iteración {initial_idx + idx+1}")
            plt.xlabel("Fecha")
            plt.ylabel("Valor")
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.gcf().autofmt_xdate()  # Rotar las etiquetas de fecha
            plt.grid(True)

            plt.legend()

            # Guardar la figura
            plot_filename = os.path.join(save_dir, f"iteracion_{initial_idx + idx+1}.png")
            plt.savefig(plot_filename)
            plt.close()  # Cerrar la figura para liberar memoria



            # # Añadir barras verticales discontinuas cada 25 valores y etiquetarlas
            # dates = test["Date"].to_list()

            #         # Gráfica de las predicciones y los valores reales
            # plt.plot(dates, preds_flat, color="r", label="Predicciones", marker="o", linestyle="--")
            # plt.plot(dates, real_vals, color="g", label="Valores Reales", marker="x", linestyle="-")

            # num_barras = len(dates) // params["pred_len"]
            # for i in range(1, num_barras + 1):
            #     pos = i * 25
            #     if pos < len(dates):  # Asegurarse de no exceder el rango
            #         plt.axvline(x=dates[pos], color="b", linestyle="--", linewidth=0.8)
            #         plt.text(dates[pos], max(preds_flat), f'Pred {i}', rotation=90, ha='center', color="blue", fontsize=8)

            # # Personalización del gráfico
            # plt.title(f"Predicciones vs Valores Reales - Iteración {idx+1}")
            # plt.xlabel("Fecha")
            # plt.ylabel("Valor")
            # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            # plt.gcf().autofmt_xdate()  # Rotar las etiquetas de fecha
            # plt.grid(True)
            # plt.legend()


            # # Guardar la figura
            # plot_filename = os.path.join(save_dir, f"iteracion_{idx+1}.png")
            # plt.savefig(plot_filename)
            # plt.close()  # Cerrar la figura para liberar memoria

        except:
            tft_predict(tft, val_dataloader)

    print(
            f"Mejores hiperparámetros: {best_params} con una pérdida de validación de {best_val_loss:.4f}"
        )

    

    return best_model, best_params, best_val_loss


def exhaustive_hyperparameter_search(
    data,
    train,
    train_dataloader,
    val_dataloader,
    test,
    param_grid,
    max_epochs=1,
    save_dir="plots",
    csv_file="resultados.csv",
):
    best_val_loss = float("inf")
    best_params = None
    best_model = None

    # Crear combinaciones de parámetros
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Crear el directorio para guardar las imágenes si no existe
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Si el archivo CSV no existe, crear uno nuevo con encabezados
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Fecha",
                    "ID",
                    "Gradient Clip Val",
                    "Hidden Size",
                    "Dropout",
                    "Hidden Continuous Size",
                    "Attention Head Size",
                    "Learning Rate",
                    "Loss Function",
                    "Epochs",
                    "Val Loss",
                    "Test MAE",
                    "Test MAPE",
                    "Test RMSE",
                    "Training Time (s)",
                ]
            )

    for idx, params in enumerate(combinations):
        print(f"Probando combinación {idx+1}/{len(combinations)}: {params}")

        # Medir el tiempo de inicio del entrenamiento
        start_time = datetime.now()

        # Entrenar el modelo con la combinación actual de hiperparámetros
        tft, val_loss = tft_trainer(
            train, train_dataloader, val_dataloader, max_epochs=max_epochs, **params
        )

        # Medir el tiempo de finalización del entrenamiento
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # Generar predicciones en los datos de prueba usando los parámetros actuales
        preds = make_preds(
            train=data,
            test=test,
            model=tft,
            encoder_lenght=params["n_prev_len"],
            test_lenght=params["test_len"],
            pred_lenght=params["pred_len"],
            quantiles=True if isinstance(params["loss"], QuantileLoss) else False,
        )

        # Calcular métricas de error en test
        preds_flat = [item for sublist in preds for item in sublist]
        real_vals = test["target"].to_list()
        test_mae = mean_absolute_error(real_vals, preds_flat)
        test_mape = mean_absolute_percentage_error(real_vals, preds_flat)
        test_rmse = mean_squared_error(real_vals, preds_flat, squared=False)

        # Verificar si la pérdida es la mejor hasta ahora
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            best_model = tft
            print(f"Nueva mejor combinación encontrada: {params} con pérdida {val_loss:.4f}")

        # Guardar los resultados de la iteración en el archivo CSV
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    idx + 1,
                    params["gradient_clip_val"],
                    params["hidden_size"],
                    params["dropout"],
                    params["hidden_continuous_size"],
                    params["attention_head_size"],
                    params["learning_rate"],
                    params["loss"].__class__.__name__,
                    max_epochs,
                    val_loss,
                    test_mae,
                    test_mape,
                    test_rmse,
                    training_time,
                ]
            )

        # Crear y guardar el gráfico de esta iteración
        dates = test["fechaHora"].to_list()
        plt.figure(figsize=(10, 6))
        plt.plot(dates, preds_flat, color="r", label="Predicciones", marker="o", linestyle="--")
        plt.plot(dates, real_vals, color="g", label="Valores Reales", marker="x", linestyle="-")
        plt.title(f"Predicciones vs Valores Reales - Iteración {idx+1}")
        plt.xlabel("Fecha")
        plt.ylabel("Valor")
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.gcf().autofmt_xdate()  # Rotar las etiquetas de fecha
        plt.grid(True)
        plt.legend()

        # Guardar la figura
        plot_filename = os.path.join(save_dir, f"iteracion_{idx+1}.png")
        plt.savefig(plot_filename)
        plt.close()  # Cerrar la figura para liberar memoria

    print(
        f"Mejores hiperparámetros: {best_params} con una pérdida de validación de {best_val_loss:.4f}"
    )

    return best_model, best_params, best_val_loss


def add_sma(df, period):
    df[f"SMA_{period}"] = df["target"].rolling(window=period).mean()
    return df


def add_ema(df, period):
    df[f"EMA_{period}"] = df["target"].ewm(span=period, adjust=False).mean()
    return df


def add_rsi(df, period=14):
    delta = df["target"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df, short_period=12, long_period=26, signal_period=9):
    short_ema = df["target"].ewm(span=short_period, adjust=False).mean()
    long_ema = df["target"].ewm(span=long_period, adjust=False).mean()
    df["MACD"] = short_ema - long_ema
    df["Signal_Line"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
    return df


def add_bollinger_bands(df, period=20):
    sma = df["target"].rolling(window=period).mean()
    std = df["target"].rolling(window=period).std()

    df[f"Bollinger_Upper_{period}"] = sma + (std * 2)
    df[f"Bollinger_Lower_{period}"] = sma - (std * 2)
    return df


def add_cci(df, period=20):
    typical_price = df["target"]
    sma = typical_price.rolling(window=period).mean()
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    df[f"CCI_{period}"] = (typical_price - sma) / (0.015 * mean_deviation)
    return df


def add_atr(df, period=14):
    df["High_Low"] = df["target"].rolling(window=1).max() - df["target"].rolling(window=1).min()
    df["High_Close"] = abs(df["target"] - df["target"].shift(1))
    df["Low_Close"] = abs(df["target"].shift(1) - df["target"])
    df["True_Range"] = df[["High_Low", "High_Close", "Low_Close"]].max(axis=1)
    df[f"ATR_{period}"] = df["True_Range"].rolling(window=period).mean()
    df.drop(columns=["High_Low", "High_Close", "Low_Close", "True_Range"], inplace=True)
    return df


def add_roc(df, period=12):
    df[f"ROC_{period}"] = df["target"].pct_change(periods=period) * 100
    return df


def add_stochastic(df, period=14):
    df[f"Stochastic_{period}_K"] = 100 * (
        (df["target"] - df["target"].rolling(window=period).min())
        / (df["target"].rolling(window=period).max() - df["target"].rolling(window=period).min())
    )
    df[f"Stochastic_{period}_D"] = df[f"Stochastic_{period}_K"].rolling(window=3).mean()
    return df


def add_williams_r(df, period=14):
    high = df["target"].rolling(window=period).max()
    low = df["target"].rolling(window=period).min()
    df[f"Williams_%R_{period}"] = -100 * (high - df["target"]) / (high - low)
    return df


# def add_parabolic_sar(df, af=0.02, max_af=0.2):
#     df["SAR"] = tal.sar(high=df["target"], low=df["target"], af=af, max_af=max_af)
#     return df


# def add_ichimoku(df):
#     ichimoku = ta.ichimoku(high=df["target"], low=df["target"], close=df["target"], return_as="df")
#     df = pd.concat([df, ichimoku[0]], axis=1)
#     return df


def clean_numeric_column(column):
    return column.str.replace(".", "", regex=False).str.replace(",", ".").astype(float)


# Función para limpiar la columna de volumen
def clean_volume_column(column):
    column = column.str.replace("B", "e9", regex=False).str.replace("M", "e6", regex=False)
    return column.str.replace(".", "", regex=False).str.replace(",", ".").astype(float)


# Función para limpiar columnas de porcentaje
def clean_percentage_column(column):
    return column.str.replace("%", "", regex=False).str.replace(",", ".").astype(float)


def load_file(
    file_name: str,
    path: str,
    ftype: str = "xlsx",
    # env: str = "LOCAL",
    sep: str = ",",
    sheet_name: str = 0,
    header: int = "infer",
    index_col: str = None,
    usecols: list = None,
    dtypes: dict = None,
    **kwargs,
):
    """
    Load a file

    Parameters
    ----------
    file_name : str
        Name of the file to be loaded
    path : str
        path to file
    ftype : str, optional
        File extension, by default CSV
    sep : str, optional
        Column sep for CSV
    usecols : list, optional
        list of columns from the table to be loaded, by default None which implies all columns
    dtypes: dict, optional
        Dictionary of data types for table columns, by default None

    Returns
    -------
    Pandas Dataframe
        Dataframe loaded from file
    """

    data = None
    if ftype.lower() == "csv" or ftype.lower() == "txt":
        # Load file
        data = pd.read_csv(
            path + f"{file_name}.{ftype}",
            sep=sep,
            header=header,
            usecols=usecols,
            dtype=dtypes,
            index_col=index_col,
            on_bad_lines='skip',
            skiprows=kwargs.get("skiprows", 0),
            skipfooter=kwargs.get("skipfooter", 0),
            encoding=kwargs.get("encoding", "utf-8"),
        )

    elif ftype.lower().startswith("xls"):
        data = pd.read_excel(
            os.path.join(
                path,
                f"{file_name}.{ftype}",
            ),
            sheet_name=sheet_name,
            usecols=usecols,
            dtype=dtypes,
            skiprows=kwargs.get("skiprows", 0),
            skipfooter=kwargs.get("skipfooter", 0),
            engine=kwargs.get("engine", None),
        )

    elif ftype.lower() == "pkl":
        with open(os.path.join(path, f"{file_name}.{ftype}"), "rb") as f:  # load best model params
            data = pickle.load(f)
    # df = pd.read_excel(os.path.join(path))

    return data


def save_file(
    data,
    file_name: str,
    path: str,
    ftype: str = "xlsx",
    # env: str = "LOCAL",
    sep: str = ",",
    sheet_name: str = 0,
    header: int = "infer",
    index=False,
    usecols: list = None,
):
    """Saves dataframe

        Parameters
        ----------
        df : Dataframe
        file_name: str,
        path : str
        ftype: str = "xlsx",
        env: str = "LOCAL",
        sep: str = ";",
        sheet_name: str = 0,
        header: int = "infer",
        usecols: list = None,
        dtypes: dict = None,
        parse_dates: bool = True,
    ):

    """
    if ftype.lower() == "csv" or ftype.lower() == "txt":
        data.to_csv(
            os.path.join(
                path,
                f"{file_name}.{ftype}",
            ),
            sep=sep,
            columns=usecols,
            index=index,
        )

    elif ftype.lower().startswith("xls"):
        data.to_excel(
            os.path.join(
                path,
                f"{file_name}.{ftype}",
            ),
            # columns=usecols,
            # header=header,
            # sheet_name=sheet_name,
            index=index,
        )

    elif ftype.lower() == "pkl":
        with open(os.path.join(path, f"{file_name}.{ftype}"), "wb") as f:
            pickle.dump(data, f)

def investing_preprocessing(df):
    # Detectar el idioma de las columnas y renombrarlas si es necesario
    if "Fecha" in df.columns:
        column_map = {
            "Fecha": "Date",
            "Último": "Price",
            "Apertura": "Open",
            "Máximo": "High",
            "Mínimo": "Low",
            "Vol.": "Vol.",
            "% var.": "Change %"
        }
    elif "Date" in df.columns:
        column_map = {}  # No se necesita mapeo, ya está en inglés
    else:
        raise ValueError("El DataFrame no contiene columnas reconocidas.")

    # Renombrar columnas si es necesario
    df = df.rename(columns=column_map)
    
    # Revertir el DataFrame y resetear índice
    df = df[::-1].reset_index(drop=True)
    
    # Formatear la columna de fechas
    try:
        df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    except:
        pass
    
    # Aplicar las transformaciones numéricas
    df["Price"] = clean_numeric_column(df["Price"])
    df["Open"] = clean_numeric_column(df["Open"])
    df["High"] = clean_numeric_column(df["High"])
    df["Low"] = clean_numeric_column(df["Low"])
    
    # Limpiar columna de volumen si está disponible
    try:
        df["Vol."] = clean_volume_column(df["Vol."])
    except:
        df = df.drop(columns=["Vol."])
        print('Dataset does not contain volume data.')

    # Limpiar columna de cambio porcentual
    df["Change %"] = clean_percentage_column(df["Change %"])

    # Renombrar las columnas finales
    df = df.rename(
        columns={
            "Date": "Date",
            "Vol.": "vol",
            "Change %": "var",
            "Price": "target",
            "High": "max",
            "Low": "min",
            "Open": "open",
        }
    )
    
    df = df.reset_index(drop=True)
    return df

def add_indicators(df, ts_indicator_params, categorical_tendency_vars=False):
    # añadimos lags
    lags = buildLaggedFeatures(df, ts_indicator_params["n_lags"], ["target"])
    df = pd.concat([df, lags], axis=1)
    
    # # Apply gaussian filter to original ts.
    for sigma in ts_indicator_params["sigma_gaussian_filter"]:
        df[f'target_smoothed_{sigma}'] = gaussian_filter1d(df['target'].values, sigma=sigma)

    for i in ts_indicator_params["moving_average_windows"]:
        df = add_sma(df, period=i)
        df = add_ema(df, period=i)

    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_atr(df)

    # Agregar CCI con diferentes períodos
    for i in [10,20]:
        df = add_cci(df, period=i)

    # Agregar ROC con diferentes períodos
    for i in [10, 14, 20]:
        df = add_roc(df, period=i)

    # Agregar Stochastic Oscillator con diferentes períodos
    df = add_stochastic(df)

    # Agregar Williams %R con diferentes períodos
    df = add_williams_r(df)

    # Suponiendo que ya has agregado los indicadores a df

    # Crear variables binarias para identificar tendencias
    # 1. Tendencia alcista/bajista usando medias móviles
    df['bullish_sma_50_200'] = (df['SMA_50'] > df['SMA_200']).astype(int)  # 1 si la SMA de 50 > SMA de 200
    df['bearish_sma_50_200'] = (df['SMA_50'] < df['SMA_200']).astype(int)  # 1 si la SMA de 50 < SMA de 200

    # 2. Tendencia alcista/bajista usando RSI
    df['bullish_rsi'] = (df['RSI_14'] < 30).astype(int)  # 1 si RSI es menor que 30 (sobreventa)
    df['bearish_rsi'] = (df['RSI_14'] > 70).astype(int)  # 1 si RSI es mayor que 70 (sobrecompra)

    # 3. Tendencia alcista/bajista usando Bandas de Bollinger
    df['bullish_bollinger'] = (df['target'] < df['Bollinger_Lower_20']).astype(int)  # 1 si el precio está por debajo de la banda inferior
    df['bearish_bollinger'] = (df['target'] > df['Bollinger_Upper_20']).astype(int)  # 1 si el precio está por encima de la banda superior

    # 4. Tendencia alcista/bajista usando MACD
    df['bullish_macd'] = (df['MACD'] > df['Signal_Line']).astype(int)  # 1 si MACD es mayor que la señal
    df['bearish_macd'] = (df['MACD'] < df['Signal_Line']).astype(int)  # 1 si MACD es menor que la señal

    # 5. Tendencia alcista/bajista usando ATR
    df['bullish_atr'] = (df['ATR_14'] > df['ATR_14'].rolling(window=14).mean()).astype(int)  # 1 si ATR actual es mayor que la media
    df['bearish_atr'] = (df['ATR_14'] < df['ATR_14'].rolling(window=14).mean()).astype(int)  # 1 si ATR actual es menor que la media

    # Ejemplo de combinación de señales para una tendencia general
    df['bullish_trend'] = ((df['bullish_sma_50_200'] + df['bullish_rsi'] + 
                            df['bullish_bollinger'] + df['bullish_macd'] + 
                            df['bullish_atr']) > 2).astype(int)

    df['bearish_trend'] = ((df['bearish_sma_50_200'] + df['bearish_rsi'] + 
                            df['bearish_bollinger'] + df['bearish_macd'] + 
                            df['bearish_atr']) >= 3).astype(int)

    binary_columns = [
        'bullish_sma_50_200', 'bearish_sma_50_200', 
        'bullish_rsi', 'bearish_rsi', 
        'bullish_bollinger', 'bearish_bollinger', 
        'bullish_macd', 'bearish_macd', 
        'bullish_atr', 'bearish_atr', 
        'bullish_trend', 'bearish_trend'
    ]

    if categorical_tendency_vars == True:
    # Convertir cada columna binaria a string
        for col in binary_columns:
            df[col] = df[col].astype(str)

    return df

def add_global_indicators(df, PIB_relevant_countries, date_start, date_end):
    # Cargamos los datos de la tasa de interes (Federal Funds Effective Rate) de estados unidos. https://fred.stlouisfed.org/series/FEDFUNDS
    FedFundsRate_df = load_file(file_name="FedFundsEffRate", path="./data/", ftype="csv")
    FedFundsRate_df = FedFundsRate_df.rename(columns={"DATE": "Date"})
    FedFundsRate_df["Date"] = pd.to_datetime(FedFundsRate_df["Date"], format="%Y-%m-%d")

    # Cargamos los datos del indice VIX. https://fred.stlouisfed.org/series/VIXCLS
    VIX_df = load_file(file_name="VIX_FRED", path="./data/", ftype="csv")
    VIX_df = VIX_df.rename(columns={"DATE": "Date", "VIXCLS": "VIX"})
    VIX_df["Date"] = pd.to_datetime(VIX_df["Date"], format="%Y-%m-%d")

    # Cargamos los datos del indice VIX Europeo. https://es.investing.com/indices/stoxx-50-volatility-vstoxx-eur
    EUVIX_df = load_file(file_name="VSTOXX", path="./data/", ftype="csv")
    EUVIX_df = EUVIX_df.rename(columns={"target": "EUVIX"})
    EUVIX_df = EUVIX_df[['Date', 'Price']]
    EUVIX_df["Date"] = pd.to_datetime(EUVIX_df["Date"], format="%m/%d/%Y")
    EUVIX_df = EUVIX_df.rename(columns={"Price": "EUVIX"})


    # Cargamos los datos del PIB de los paises seleccionados
    worldPIB = load_file(file_name="worldPIBdata", path="./data/", ftype="xls", skiprows=3)
    worldPIB = worldPIB.iloc[:, [0, 1] + list(range(50, len(worldPIB.columns)))]
    # df = df.iloc[1:, :]

    worldPIB=worldPIB.set_index('Country Code').loc[PIB_relevant_countries].T
    worldPIB = worldPIB.drop(index='Country Name')
    worldPIB.index.name = 'Date'
    worldPIB.reset_index(inplace=True)
    worldPIB["Date"] = pd.to_datetime(worldPIB["Date"], format="%Y")
    worldPIB.columns.name = ''
    worldPIB = worldPIB.rename(columns={col: f"PIB_{col}" for col in worldPIB.columns[1:]})


    # Cargamos los datos de AAII (American Association of Individual Investors)
    AAII_df=load_file(file_name="IIAA_sentiment", path="./data/", ftype="xls", usecols=range(0,4), **{"skiprows":3, "skipfooter":203})
    AAII_df["Date"] = pd.to_datetime(AAII_df["Date"], format="%Y-%m-%d %H:%M:%S")

    # Loas AAII stock sentiment historic data.
    AAII_df = AAII_df.rename(
        columns={
            "Bullish": "AAII_Bullish",
            "Neutral": "AAII_Neutral",
            "Bearish": "AAII_Bearish",})

    # Get selected time range for training
    if date_end:
        df = df[df["Date"] <= date_end]
        AAII_df = AAII_df[AAII_df["Date"] < date_end]


    if date_start:
        df = df[df["Date"] >= date_start]
        AAII_df = AAII_df[AAII_df["Date"] >= date_start]

    # Creo que es mas eficiente asi que eliminar con el merge

    AAII_df = AAII_df.reset_index(drop=True)

    # Merge dataframes to create training df.
    df = df.merge(AAII_df, on="Date", how="left")
    df = df.merge(FedFundsRate_df, on="Date", how="left")
    df = df.merge(VIX_df, on="Date", how="left")
    df = df.merge(EUVIX_df, on="Date", how="left")

    df['year'] = pd.to_datetime(df['Date']).dt.year

    # Crear una columna temporal 'year' en df2 convirtiendo 'date' a enteros
    # Extraer el año directamente de la columna 'Date' en worldPIB
    worldPIB['year'] = pd.to_datetime(worldPIB['Date']).dt.year

    # Realizar el merge usando la columna temporal 'year'
    df = pd.merge(df, worldPIB, on='year', how='left')
    df = df.rename(columns={'Date_x': 'Date'})
    df.drop(columns=['Date_y'], inplace=True)
    # Eliminar la columna temporal 'year' después del merge
    df = df.drop(columns=['year'])

    df = df.ffill()
    df = df.bfill()

    return df

def create_combined_ts_df(target_file, exog_files):
    """
    Combines a target time series with multiple exogenous time series into a single DataFrame.

    This function takes a target file containing the main time series and a list of files 
    with exogenous time series. It preprocesses all input files, renames the target column 
    of each exogenous series, and merges them with the target series based on the 'Date' column.

    Parameters:
    ----------
    target_file : str
        The name of the file containing the target time series (excluding the file extension).
    exog_files : list of str
        A list of file names containing the exogenous time series (excluding file extensions).

    Returns:
    -------
    pandas.DataFrame
        A DataFrame that combines the target time series and all provided exogenous series. 
        The resulting DataFrame includes the 'Date' column, the target series, and each 
        exogenous series as separate columns.

    Example:
    -------
    target_file = 'S&P500'
    exog_files = ['Nasdaq', 'IBEX35', 'EUStoxx50', 'DowJones', 'BTC']
    df = combine_target_and_exog_ts(target_file, exog_files)

    Notes:
    -----
    - The function assumes all input files are in the './data/' directory and are in CSV format.
    - The preprocessing steps applied are handled by `load_file` and `investing_preprocessing` functions.
    - The merge is performed on the 'Date' column using a left join, ensuring that all dates in the 
      target series are preserved, even if some exogenous series are missing values for certain dates.
    """

    exog_dfs = []
    for exog_file in exog_files:
        df = load_file(file_name=exog_file, path="./data/", ftype="csv")
        df = investing_preprocessing(df)
        df = df.rename(columns={"target": f"exog_{exog_file}"})
        exog_dfs.append(df)

    target_df = load_file(file_name=target_file, path="./data/", ftype="csv")
    target_df = investing_preprocessing(target_df)

    result_df = target_df.copy()  # Copy the base DataFrame to avoid modifying the original
    for exog_df in exog_dfs:
        target_column = exog_df.columns[1]  # The 2nd column is treated as the exogenous 'target'
        # Merge on the 'Date' column
        result_df = result_df.merge(exog_df[['Date', target_column]], on='Date', how='left')
    return result_df
