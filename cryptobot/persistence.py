"""Mixin para persistencia: guardar/cargar estado y trade history."""

import os

import joblib
import pandas as pd


class PersistenceMixin:
    """Métodos de persistencia: save(), load(), trade_history()."""

    def save(self, name: str = None, path: str = "./cryptobot_saves") -> None:
        """
        Guarda el estado completo del bot a disco.

        Guarda: modelo entrenado, configuración, régimen, features,
        historial de trades, y métricas. NO guarda datos OHLCV crudos
        (se pueden re-descargar con fetch_data).

        Parameters
        ----------
        name : str, optional
            Nombre del archivo (sin extensión).
            Default: {symbol}_{timeframe}_{strategy}.
        path : str, default "./cryptobot_saves"
            Directorio donde guardar. En Colab usa
            "/content/drive/MyDrive/" para persistencia.

        Examples
        --------
        >>> bot.save()
        >>> bot.save("mi_bot_v1")
        >>> bot.save("mi_bot_v1", path="/content/drive/MyDrive/bots/")
        """
        if name is None:
            strategy = self.selected_strategy or "no_strategy"
            name = f"{self.symbol}_{self.timeframe}_{strategy}"

        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(path, f"{name}.pkl")

        # Estado a guardar (NO datos OHLCV — se re-descargan)
        state = {
            # Config
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "exchange_id": self.exchange_id,
            "max_position_pct": self.max_position_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            # Modelo
            "model": self.model,
            "model_name": self.model_name,
            "model_metrics": self.model_metrics,
            "model_comparison": self.model_comparison,
            "_feature_cols": getattr(self, "_feature_cols", None),
            # Régimen
            "regime": self.regime,
            "regime_model": self.regime_model,
            "regime_probabilities": self.regime_probabilities,
            # Estrategia
            "selected_strategy": self.selected_strategy,
            # Señales
            "signals": self.signals,
            # Trades
            "trades": self.trades,
        }

        joblib.dump(state, filepath)
        file_size = os.path.getsize(filepath) / 1024

        print(f"💾 Bot guardado: {filepath} ({file_size:.1f} KB)")
        print(f"   Modelo: {self.model_name or 'ninguno'}")
        print(f"   Régimen: {self.regime or 'no detectado'}")
        print(f"   Estrategia: {self.selected_strategy or 'ninguna'}")
        print(f"   Trades: {len(self.trades)}")

    def load(self, name: str = None, path: str = "./cryptobot_saves") -> "PersistenceMixin":
        """
        Carga estado previamente guardado.

        Parameters
        ----------
        name : str, optional
            Nombre del archivo (sin extensión).
            Default: {symbol}_{timeframe}_{strategy}.
        path : str, default "./cryptobot_saves"
            Directorio donde buscar.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        Después de cargar, aún necesitas ejecutar fetch_data() para
        obtener datos actualizados. El modelo y configuración se
        restauran automáticamente.

        Examples
        --------
        >>> bot = CryptoBot()
        >>> bot.load("mi_bot_v1")
        >>> bot.fetch_data()  # datos frescos
        >>> bot.get_signals()  # usa modelo cargado
        """
        if name is None:
            strategy = self.selected_strategy or "no_strategy"
            name = f"{self.symbol}_{self.timeframe}_{strategy}"

        filepath = os.path.join(path, f"{name}.pkl")

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"❌ No se encontró: {filepath}\n"
                f"   Archivos disponibles: {os.listdir(path) if os.path.exists(path) else '(directorio no existe)'}"
            )

        state = joblib.load(filepath)

        # Restaurar config
        self.symbol = state["symbol"]
        self._pair = f"{self.symbol}/USDT"
        self.timeframe = state["timeframe"]
        self.exchange_id = state["exchange_id"]
        self.max_position_pct = state["max_position_pct"]
        self.stop_loss_pct = state["stop_loss_pct"]
        self.take_profit_pct = state["take_profit_pct"]

        # Restaurar modelo
        self.model = state["model"]
        self.model_name = state["model_name"]
        self.model_metrics = state["model_metrics"]
        self.model_comparison = state["model_comparison"]
        if state.get("_feature_cols") is not None:
            self._feature_cols = state["_feature_cols"]

        # Restaurar régimen
        self.regime = state["regime"]
        self.regime_model = state["regime_model"]
        self.regime_probabilities = state["regime_probabilities"]

        # Restaurar estrategia
        self.selected_strategy = state["selected_strategy"]

        # Restaurar señales
        self.signals = state["signals"]

        # Restaurar trades
        self.trades = state["trades"]

        # Reinicializar exchange con la config cargada
        self._init_exchange()

        print(f"📂 Bot cargado: {filepath}")
        print(f"   Símbolo: {self.symbol} | Timeframe: {self.timeframe}")
        print(f"   Modelo: {self.model_name or 'ninguno'}")
        print(f"   Régimen: {self.regime or 'no detectado'}")
        print(f"   Estrategia: {self.selected_strategy or 'ninguna'}")
        print(f"   Trades: {len(self.trades)}")
        print(f"\n   ℹ️  Ejecuta fetch_data() para obtener datos actualizados.")

        return self

    def trade_history(self) -> pd.DataFrame:
        """
        Retorna historial de trades como DataFrame.

        Returns
        -------
        pd.DataFrame
            Columnas: timestamp, type (BUY/SELL), symbol, amount,
            price, stop_loss, take_profit, pnl, status.

        Notes
        -----
        Exportable con: bot.trade_history().to_csv("mis_trades.csv")
        """
        if not self.trades:
            print("📭 No hay trades registrados aún.")
            return pd.DataFrame()

        return pd.DataFrame(self.trades)
