"""Mixin para feature engineering: indicadores técnicos."""

import ta
import pandas as pd


class FeaturesMixin:
    """Métodos de feature engineering: create_features()."""

    def create_features(self, mode: str = "core") -> "FeaturesMixin":
        """
        Agrega indicadores técnicos al DataFrame.

        Dos modos disponibles:
        - "core": ~10 indicadores clave que se cubrieron en el curso.
          Ideal para exploración y comprensión.
        - "full": 86+ indicadores via ta.add_all_ta_features().
          Ideal para training de modelos donde feature importance
          determina qué indicadores son relevantes.

        Parameters
        ----------
        mode : str, default "core"
            "core" para indicadores esenciales, "full" para todos.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Core Features
        -------------
        - SMA_20, SMA_50 : Medias móviles simples
        - RSI_14 : Relative Strength Index
        - MACD, MACD_signal : Moving Average Convergence Divergence
        - BB_upper, BB_lower : Bollinger Bands
        - ATR_14 : Average True Range (volatilidad)
        - volume_change : Cambio porcentual del volumen
        - returns : Retorno porcentual diario
        - volatility_20 : Volatilidad rolling 20 períodos

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado fetch_data() previamente.
        ValueError
            Si mode no es "core" o "full".
        """
        self._require_data()

        df = self.data.copy()
        original_rows = len(df)

        if mode == "core":
            # Medias móviles
            df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
            df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)

            # RSI
            df["RSI_14"] = ta.momentum.rsi(df["Close"], window=14)

            # MACD
            df["MACD"] = ta.trend.macd(df["Close"])
            df["MACD_signal"] = ta.trend.macd_signal(df["Close"])

            # Bollinger Bands
            df["BB_upper"] = ta.volatility.bollinger_hband(df["Close"])
            df["BB_lower"] = ta.volatility.bollinger_lband(df["Close"])

            # ATR
            df["ATR_14"] = ta.volatility.average_true_range(
                df["High"], df["Low"], df["Close"], window=14
            )

            # Indicadores manuales
            df["volume_change"] = df["Volume"].pct_change()
            df["returns"] = df["Close"].pct_change()
            df["volatility_20"] = df["returns"].rolling(window=20).std()

            df.dropna(inplace=True)
            n_features = 11

        elif mode == "full":
            df = ta.add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume"
            )
            # add_all_ta_features no genera estos
            df["returns"] = df["Close"].pct_change()
            df["volatility_20"] = df["returns"].rolling(window=20).std()

            # ffill para indicadores con NaN por diseño (e.g. PSAR up/down)
            df.ffill(inplace=True)
            df.dropna(inplace=True)
            n_features = len(df.columns) - 5  # descontar OHLCV originales

        else:
            raise ValueError(f"mode debe ser 'core' o 'full', recibido: '{mode}'")

        self.features = df

        print(f"🔧 Features creados ({mode}): {n_features} indicadores")
        print(f"   Registros: {len(df)} (de {original_rows} originales)")

        return self
