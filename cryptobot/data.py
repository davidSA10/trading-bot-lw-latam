"""Mixin para data pipeline: descarga y resumen de datos OHLCV."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


class DataMixin:
    """Métodos de data pipeline: fetch_data() y summary()."""

    def fetch_data(
        self,
        last_n: int = 200,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> "DataMixin":
        """
        Obtiene datos OHLCV del exchange via CCXT.

        Soporta dos modos:
        - Por número de velas: bot.fetch_data(last_n=200)
        - Por rango de fechas: bot.fetch_data(start="2024-01-01", end="2024-12-31")

        La ventana temporal se calcula automáticamente según el timeframe:
        - last_n=200 con timeframe="1h" → últimas 200 horas (~8 días)
        - last_n=200 con timeframe="4h" → últimas 800 horas (~33 días)
        - last_n=200 con timeframe="1d" → últimos 200 días (~6.5 meses)

        Las columnas del DataFrame siguen el formato requerido
        por backtesting.py: Open, High, Low, Close, Volume.

        Parameters
        ----------
        last_n : int, default 200
            Número de velas hacia atrás desde hoy. Se ignora si start/end están definidos.
        start : str, optional
            Fecha de inicio en formato "YYYY-MM-DD".
        end : str, optional
            Fecha de fin en formato "YYYY-MM-DD". Default: hoy.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Examples
        --------
        >>> bot.fetch_data()
        >>> bot.fetch_data(last_n=500)
        >>> bot.fetch_data(start="2024-01-01", end="2024-06-30")
        """
        # ── Calcular timestamps ────────────────────────
        if start is not None:
            since_timestamp = int(
                datetime.strptime(start, "%Y-%m-%d").timestamp() * 1000
            )
            if end is not None:
                end_timestamp = int(
                    datetime.strptime(end, "%Y-%m-%d").timestamp() * 1000
                )
            else:
                end_timestamp = int(datetime.now().timestamp() * 1000)
        else:
            end_timestamp = int(datetime.now().timestamp() * 1000)
            tf_hours = {"1h": 1, "4h": 4, "1d": 24}
            hours = last_n * tf_hours[self.timeframe]
            since_timestamp = int(
                (datetime.now() - timedelta(hours=hours)).timestamp() * 1000
            )

        # ── Fetch paginado ────────────────────────────
        all_candles = []
        since = since_timestamp
        limit = 1000  # Bybit v5 soporta hasta 1000 por request

        while since < end_timestamp:
            batch = self._exchange.fetch_ohlcv(
                self._pair, self.timeframe, since=since, limit=limit
            )
            if not batch:
                break
            # Filtrar candles que excedan end_timestamp
            batch = [c for c in batch if c[0] <= end_timestamp]
            all_candles.extend(batch)
            if len(batch) < limit:
                break
            since = batch[-1][0] + 1

        # ── Caso sin datos ────────────────────────────
        if not all_candles:
            print(f"❌ No se obtuvieron datos para {self._pair}")
            return self

        # ── Construir DataFrame ───────────────────────
        df = pd.DataFrame(
            all_candles, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
        df = df.set_index("Date").drop(columns=["Timestamp"])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        self.data = df

        # ── Resumen ───────────────────────────────────
        print(f"📊 {self._pair} | {self.timeframe}")
        print(f"   Período: {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
        print(f"   Registros: {len(df)}")
        print(f"   Precio actual: ${df['Close'].iloc[-1]:,.2f}")

        return self

    def summary(self) -> None:
        """
        Muestra resumen del dataset cargado.

        Incluye: rango de fechas, número de registros, precio actual,
        cambio porcentual, high/low del período, y estadísticas básicas.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado fetch_data() previamente.
        """
        self._require_data()

        df = self.data
        first_close = df["Close"].iloc[0]
        last_close = df["Close"].iloc[-1]
        change_pct = (last_close - first_close) / first_close * 100
        period_high = df["High"].max()
        period_low = df["Low"].min()

        print(f"\n{'═' * 50}")
        print(f"  📊 Resumen: {self._pair} | {self.timeframe}")
        print(f"{'═' * 50}")
        print(f"  Período:    {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}")
        print(f"  Registros:  {len(df)}")
        print(f"  Precio actual: ${last_close:,.2f}")
        print(f"  Cambio:     {change_pct:+.2f}%")
        print(f"  High:       ${period_high:,.2f}")
        print(f"  Low:        ${period_low:,.2f}")
        print(f"{'─' * 50}")
        print(df.describe())
