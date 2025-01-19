import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


class TradingEnv(Env):

    def __init__(self, data, perc_to_inv=0.1, initial_balance=10000):
        # Dataset y parámetros iniciales
        self.data = data
        self.perc_to_inv = perc_to_inv
        self.initial_balance = initial_balance

        # Espacios de acción: 0 = Mantener, 1 = Comprar, 2 = Vender
        self.action_space = Discrete(3)

        # Espacios de observación: Estado del mercado (excluyendo columnas irrelevantes)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(data.shape[1],), dtype=np.float32
        )

        # Variables internas
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # Número de acciones compradas
        self.current_step = 0
        self.done = False

        return self._get_observation()

    def step(self, action):
        # Guardar estado inicial
        prev_balance = self.balance  # Dinero disponible antes de la acción.
        prev_position_value = (
            self.position * self.data["target"].iloc[self.current_step]
        )  # Valor de las acciones poseídas al precio actual.
        prev_portfolio_value = (
            prev_balance + prev_position_value
        )  # Suma del balance y el valor de las acciones

        # Ejecutar acción
        current_price = self.data["target"].iloc[self.current_step]
        if action == 1:  # Comprar
            amount_to_buy = min(self.balance, self.perc_to_inv * self.balance)
            self.position += amount_to_buy / current_price
            self.balance -= amount_to_buy
        elif action == 2:  # Vender
            amount_to_sell = min(self.position, self.perc_to_inv * self.position)
            self.balance += amount_to_sell * current_price
            self.position -= amount_to_sell

        # Calcular recompensa
        position_value = self.position * current_price
        portfolio_value = self.balance + position_value
        # reward = portfolio_value - prev_portfolio_value
        reward = (portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8)

        # Actualizar estado
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        # Retorna el estado actual (excluye columnas no relevantes)
        obs = self.data.iloc[self.current_step]
        # obs = (obs - obs.mean()) / (obs.std() + 1e-8)
        return obs.values.astype(np.float32)


def eval_trading_bot4g(data, model):
    # Evaluar manualmente en el entorno de validación
    val_env = TradingEnv(data)
    obs = val_env.reset()
    total_reward = 0
    portfolio_values = []

    while True:
        # Predecir la acción usando el modelo entrenado
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = val_env.step(action)

        total_reward += reward  # Sumar la recompensa acumulada
        portfolio_values.append(
            val_env.balance + val_env.position * val_env.data["target"].iloc[val_env.current_step]
        )

        if done:
            break

    rentabilidad_acumulada = (portfolio_values[-1] / portfolio_values[0]) - 1
    # Mostrar resultados de validación
    print(f"Recompensa acumulada en validación: {total_reward}")
    print(f"Rentabilidad acumulada: {rentabilidad_acumulada:.3%}")

    plt.plot(portfolio_values, label="Portafolio")
    plt.title("Evolución del Portafolio en Validación")
    plt.xlabel("Días")
    plt.ylabel("Valor del Portafolio")
    plt.legend()
    plt.show()

    return total_reward, rentabilidad_acumulada, portfolio_values


def eval_trading_bot(data, model, norm=False, scaler=None):
    """
    Evaluar el bot de trading en un conjunto de datos, con soporte para datos normalizados o sin normalizar.

    Args:
        data (pd.DataFrame): Conjunto de datos (puede estar normalizado o no).
        model (stable_baselines3.PPO): Modelo entrenado de PPO.
        norm (bool): Si es True, se evalúa con datos normalizados.
        scaler (sklearn.preprocessing.StandardScaler, optional): Escalador usado para normalizar/desnormalizar datos.

    Returns:
        total_reward (float): Recompensa total acumulada.
        rentabilidad_acumulada (float): Rentabilidad acumulada.
        portfolio_values (list): Evolución del valor del portafolio.
    """
    # Crear el entorno con el conjunto de datos proporcionado
    env = TradingEnv(data)
    obs = env.reset()
    total_reward = 0
    portfolio_values = []

    while True:
        # Predecir la acción usando el modelo entrenado
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        # Sumar la recompensa acumulada
        total_reward += reward

        # Calcular el valor del portafolio (normalizado o no según el caso)
        portfolio_value = env.balance + env.position * env.data["target"].iloc[env.current_step]
        portfolio_values.append(portfolio_value)

        if done:
            break

    # Calcular rentabilidad acumulada (normalizada o no)
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    rentabilidad_acumulada = (final_value / initial_value) - 1

    # Si los datos están normalizados, desnormalizar para interpretación
    if norm and scaler is not None:
        # Desnormalizar el valor del portafolio
        portfolio_values = scaler.inverse_transform([[val] for val in portfolio_values]).flatten()
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        rentabilidad_acumulada = (final_value / initial_value) - 1

    # Mostrar resultados de evaluación
    print(f"Recompensa acumulada: {total_reward}")
    print(f"Rentabilidad acumulada: {rentabilidad_acumulada:.3%}")

    # Graficar la evolución del portafolio
    plt.plot(portfolio_values, label="Portafolio (Desnormalizado)" if norm else "Portafolio")
    plt.title("Evolución del Portafolio")
    plt.xlabel("Días")
    plt.ylabel("Valor del Portafolio")
    plt.legend()
    plt.show()

    return total_reward, rentabilidad_acumulada, portfolio_values
