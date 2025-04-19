from AlgorithmImports import *

class CHPCHMomentumStrategy(QCAlgorithm):
    def Initialize(self):
        """ Configuración inicial del algoritmo """
        self.SetStartDate(2023, 1, 1)  # Fecha de inicio
        self.SetEndDate(2024, 1, 1)    # Fecha de fin
        self.SetCash(100000)           # Capital inicial
        
        # Pares de divisas a operar
        self.symbols = [
            self.AddForex("GBPUSD", Resolution.Minute, Market.Oanda).Symbol,
            self.AddForex("EURUSD", Resolution.Minute, Market.Oanda).Symbol,
            self.AddForex("USDCAD", Resolution.Minute, Market.Oanda).Symbol
        ]
        
        # Períodos de medias móviles
        self.short_ema_period = 20
        self.long_ema_period = 50

        # Configurar indicadores por cada activo
        self.data = {}
        for symbol in self.symbols:
            self.data[symbol] = SymbolData(self, symbol, self.short_ema_period, self.long_ema_period)

    def OnData(self, data):
        """ Lógica de trading en cada nuevo tick de mercado """
        for symbol in self.symbols:
            symbol_data = self.data[symbol]

            if not symbol_data.IsReady():
                continue  # Si no hay suficientes datos, pasamos al siguiente activo

            # Confirmar tendencia bajista
            if not symbol_data.IsDowntrend():
                continue

            # Buscar cambio estructural (CHPCH)
            if not symbol_data.IdentifyCHPCH():
                continue

            # Confirmar velas grandes y vacíos de liquidez
            if not symbol_data.ValidateMomentum():
                continue

            # Identificar Order Block y calcular retroceso de Fibonacci
            entry_price = symbol_data.CalculateFibonacciEntry()
            if entry_price is not None:
                continue

            # Configurar stop loss y take profit
            sl, tp = symbol_data.GetRiskManagementLevels(entry_price)
            
            # Ejecutar orden de compra
            self.MarketOrder(symbol, 1000)  # Tamaño de posición ajustable
            self.Debug(f"Compra en {symbol} - Entry: {entry_price}, SL: {sl}, TP: {tp}")


class SymbolData:
    """ Clase para manejar datos de cada símbolo individualmente """
    def __init__(self, algorithm, symbol, short_ema_period, long_ema_period):
        self.algorithm = algorithm
        self.symbol = symbol
        self.short_ema = algorithm.EMA(symbol, short_ema_period, Resolution.Minute)
        self.long_ema = algorithm.EMA(symbol, long_ema_period, Resolution.Minute)
        self.history = []

    def IsReady(self):
        """ Verifica si las EMAs tienen suficientes datos """
        return self.short_ema.IsReady and self.long_ema.IsReady

    def IsDowntrend(self):
        """ Confirma que la tendencia es bajista """
        return self.short_ema.Current.Value < self.long_ema.Current.Value

    def IdentifyCHPCH(self):
        """ Identifica un cambio estructural (mínimo más alto) """
        if len(self.history) < 3:
            return False

        lows = [bar.Low for bar in self.history[-3:]]
        return lows[1] > lows[0] and lows[1] > lows[2]

    def ValidateMomentum(self):
        """ Confirma la presencia de velas grandes y vacíos de liquidez """
        if len(self.history) < 3:
            return False
        
        candle1 = self.history[-3]
        candle2 = self.history[-2]
        candle3 = self.history[-1]

        # Validar vela grande
        avg_body_size = sum(abs(bar.Close - bar.Open) for bar in self.history[-10:]) / 10
        if abs(candle3.Close - candle3.Open) < 1.5 * avg_body_size:
            return False

        # Validar vacío de liquidez
        gap = (candle1.Close - candle1.Open) + (candle3.Close - candle3.Open)
        return gap > abs(candle2.Close - candle2.Open)

    def CalculateFibonacciEntry(self):
        """ Calcula el nivel de entrada basado en retroceso de Fibonacci """
        if len(self.history) < 5:
            return None

        recent_low = min(bar.Low for bar in self.history[-5:])
        recent_high = max(bar.High for bar in self.history[-5:])

        return recent_high - (recent_high - recent_low) * 0.8

    def GetRiskManagementLevels(self, entry_price):
        """ Define el Stop Loss y Take Profit con una relación 1:2 """
        recent_low = min(bar.Low for bar in self.history[-5:])
        sl = recent_low
        tp = entry_price + (entry_price - sl) * 2
        return sl, tp

    def UpdateHistory(self, bar):
        """ Almacena las últimas velas para análisis """
        self.history.append(bar)
        if len(self.history) > 20:
            self.history.pop(0)
