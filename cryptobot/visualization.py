"""Mixin para visualización: gráficos de precio, señales y performance."""

import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import (
    CHART_HEIGHT_MAIN,
    CHART_HEIGHT_SCAN,
    CHART_HEIGHT_SECONDARY,
    CHART_ROW_HEIGHTS,
    SCANNER_LAST_N,
    SCANNER_SYMBOLS,
)
from .constants import COLOR_PALETTE


class VisualizationMixin:
    """Métodos de visualización: plot_price(), plot_signals(), plot_performance()."""

    def plot_price(self) -> None:
        """
        Gráfico de velas japonesas (candlestick) con Plotly.

        Muestra OHLC data con volumen en panel inferior.
        Colores: paleta definida en COLOR_PALETTE.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado fetch_data() previamente.
        """
        self._require_data()
        df = self.data

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=CHART_ROW_HEIGHTS,
            vertical_spacing=0.05,
        )

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=COLOR_PALETTE["green"],
                decreasing_line_color=COLOR_PALETTE["red"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        colors = [
            COLOR_PALETTE["green"] if c >= o else COLOR_PALETTE["red"]
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["Volume"],
                marker_color=colors,
                name="Volumen",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title=f"Precio — {getattr(self, 'symbol', '')}",
            yaxis_title="Precio (USDT)",
            yaxis2_title="Volumen",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            xaxis_rangeslider_visible=False,
            height=CHART_HEIGHT_MAIN,
        )

        fig.show()

    def plot_signals(self) -> None:
        """
        Gráfico de precio con señales de trading marcadas.

        BUY señales: triángulos verdes (▲)
        SELL señales: triángulos rojos (▼)
        Overlay sobre candlestick chart.

        Raises
        ------
        RuntimeError
            Si no se han generado señales con get_signals().
        """
        self._require_signals()
        df = self.data
        signals = self.signals

        common_idx = df.index.intersection(signals.index)
        signals = signals.loc[common_idx]
        df = df.loc[common_idx]

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color=COLOR_PALETTE["green"],
                decreasing_line_color=COLOR_PALETTE["red"],
                name="OHLC",
            )
        )

        buy_idx = signals[signals == 1].index
        if len(buy_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_idx,
                    y=df.loc[buy_idx, "Low"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=COLOR_PALETTE["green"],
                    ),
                    name="BUY",
                )
            )

        sell_idx = signals[signals == -1].index
        if len(sell_idx) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_idx,
                    y=df.loc[sell_idx, "High"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color=COLOR_PALETTE["red"],
                    ),
                    name="SELL",
                )
            )

        fig.update_layout(
            title=f"Señales de Trading — {getattr(self, 'symbol', '')}",
            yaxis_title="Precio (USDT)",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            xaxis_rangeslider_visible=False,
            height=CHART_HEIGHT_SECONDARY,
        )

        fig.show()

    def plot_performance(self) -> None:
        """
        Gráfico de equity curve y drawdown.

        Panel superior: equity curve vs buy-and-hold.
        Panel inferior: drawdown.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado backtest() previamente.
        """
        if self.backtest_results is None:
            raise RuntimeError(
                "❌ No hay backtest ejecutado. Ejecuta bot.backtest() primero."
            )

        eq = self.backtest_results["_equity_curve"]
        stats = self.backtest_results

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=CHART_ROW_HEIGHTS,
            vertical_spacing=0.05,
        )

        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq["Equity"],
                mode="lines",
                name="Bot",
                line=dict(color=COLOR_PALETTE["yellow"], width=2),
            ),
            row=1,
            col=1,
        )

        close = self.data["Close"].loc[eq.index[0] : eq.index[-1]]
        initial_equity = eq["Equity"].iloc[0]
        bh = close / close.iloc[0] * initial_equity
        fig.add_trace(
            go.Scatter(
                x=bh.index,
                y=bh,
                mode="lines",
                name="Buy & Hold",
                line=dict(color="white", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=eq.index,
                y=eq["DrawdownPct"] * -100,
                mode="lines",
                fill="tozeroy",
                name="Drawdown",
                line=dict(color=COLOR_PALETTE["red"], width=1),
                fillcolor="rgba(246, 70, 93, 0.3)",
            ),
            row=2,
            col=1,
        )

        max_dd = stats.get("Max. Drawdown [%]", 0)
        final_return = stats.get("Return [%]", 0)

        fig.update_layout(
            title=f"Performance — {getattr(self, 'symbol', '')}",
            yaxis_title="Equity (USDT)",
            yaxis2_title="Drawdown (%)",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            height=CHART_HEIGHT_MAIN,
            annotations=[
                dict(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.95,
                    text=f"Retorno: {final_return:.1f}%",
                    showarrow=False,
                    font=dict(color=COLOR_PALETTE["yellow"], size=14),
                ),
                dict(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.88,
                    text=f"Max Drawdown: {max_dd:.1f}%",
                    showarrow=False,
                    font=dict(color=COLOR_PALETTE["red"], size=14),
                ),
            ],
        )

        fig.show()

    def plot_scan(self, symbols: list = None, last_n: int = SCANNER_LAST_N) -> None:
        """
        Grid visual de mini-charts con régimen coloreado.

        Muestra una línea de precio (Close) por cada criptomoneda,
        con el fondo coloreado según el régimen detectado
        (Bull=verde, Bear=rojo, Sideways=amarillo).

        Parameters
        ----------
        symbols : list of str, optional
            Lista de símbolos a graficar.
            Default: SCANNER_SYMBOLS (["BTC", "ETH", "SOL", "BNB", "XRP"])
        last_n : int, default 100
            Número de velas a descargar por símbolo.

        Examples
        --------
        >>> bot.plot_scan()
        >>> bot.plot_scan(symbols=["BTC", "ETH", "SOL"], last_n=200)
        """
        if symbols is None:
            symbols = list(SCANNER_SYMBOLS)

        if not symbols:
            print("⚠️ No hay símbolos para graficar.")
            return

        # Colores por régimen
        regime_colors = {
            "Bull": COLOR_PALETTE["green"],
            "Bear": COLOR_PALETTE["red"],
            "Sideways": COLOR_PALETTE["yellow"],
        }
        regime_emojis = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}

        # ── 1. Escanear todos los símbolos ─────────────────
        scan_results = []

        print("=" * 70)
        print("📊 SCANNER VISUAL — Generando gráficos")
        print("=" * 70)

        for symbol, temp_bot in self._scan_symbols(symbols=symbols, last_n=last_n):
            if temp_bot is not None:
                regime = temp_bot.regime
                emoji = regime_emojis.get(regime, "")
                print(f"  ✅ {symbol:<6} {regime} {emoji}")
            else:
                regime = "Error"
            scan_results.append((symbol, temp_bot, regime))

        print("=" * 70)

        # ── 2. Construir grid de subplots ──────────────────
        ncols = 2
        nrows = math.ceil(len(scan_results) / ncols)

        titles = [
            f"{symbol} — {regime} {regime_emojis.get(regime, '')}"
            for symbol, _, regime in scan_results
        ]
        # Rellenar si el número de símbolos es impar
        while len(titles) < nrows * ncols:
            titles.append("")

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=titles,
            vertical_spacing=0.08,
            horizontal_spacing=0.06,
        )

        # ── 3. Agregar traces y fondos coloreados ──────────
        for i, (symbol, temp_bot, regime) in enumerate(scan_results):
            r = i // ncols + 1
            c = i % ncols + 1

            if temp_bot is None:
                fig.add_annotation(
                    text="Error al cargar datos",
                    xref=f"x{i + 1}" if i > 0 else "x",
                    yref=f"y{i + 1}" if i > 0 else "y",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color=COLOR_PALETTE["red"], size=12),
                )
                continue

            df = temp_bot.data

            # Línea de precio Close
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["Close"],
                    mode="lines",
                    name=symbol,
                    line=dict(color="white", width=1.5),
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

            # Fondo coloreado por régimen
            fig.add_vrect(
                x0=df.index[0],
                x1=df.index[-1],
                fillcolor=regime_colors.get(regime, COLOR_PALETTE["gray"]),
                opacity=0.15,
                line_width=0,
                row=r,
                col=c,
            )

        # ── 4. Layout global ──────────────────────────────
        fig.update_layout(
            title="🔍 Scanner Visual — Regímenes de Mercado",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            height=CHART_HEIGHT_SCAN * nrows,
            showlegend=False,
        )

        fig.show()
