"""Mixin para market intelligence: régimen de mercado y estrategias."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .constants import COLOR_PALETTE, REGIME_LABELS, STRATEGY_REGISTRY


class RegimeMixin:
    """Métodos de market intelligence: detect_regime(), regime_report(), recommend_strategies(), select_strategy()."""

    def detect_regime(self, n_regimes: int = 3) -> "RegimeMixin":
        """
        Detecta el régimen de mercado actual usando Gaussian Mixture Model.

        Clasifica el mercado en regímenes usando 13 features multi-escala:
        - Retornos y tendencia en 3 timeframes (short/medium/long)
        - Volatilidad en 3 timeframes + Garman-Klass + ratio de expansión
        - Volume ratio, RSI, distancia a SMA, drawdown

        El GMM provee probabilidades suaves para cada régimen,
        no una clasificación binaria. Ejemplo: "72% Bull, 20% Sideways, 8% Bear".

        Parameters
        ----------
        n_regimes : int, default 3
            Número de regímenes a detectar.
            Default 3: Bull 🟢, Bear 🔴, Sideways 🟡.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado create_features() previamente.
        """
        self._require_features()

        # ── Constantes de ventana ────────────────────────
        SHORT, MEDIUM, LONG = 7, 21, 50
        RSI_WINDOW = 14
        STRUCTURAL_DESIRED = 200

        df = self.data.copy()
        n = len(df)

        # Ventana estructural adaptativa
        long_structural = min(STRUCTURAL_DESIRED, n - LONG)
        long_structural = max(long_structural, LONG)  # piso en 50

        # ── 1. Feature engineering: 13 indicadores ───────
        returns = df["Close"].pct_change()

        regime_features = pd.DataFrame(index=df.index)
        # Retornos
        regime_features["returns"] = returns
        # Tendencia multi-escala
        regime_features["trend_short"] = returns.rolling(SHORT).mean()
        regime_features["trend_medium"] = returns.rolling(MEDIUM).mean()
        regime_features["trend_long"] = returns.rolling(LONG).mean()
        # Volatilidad multi-escala
        regime_features["vol_short"] = returns.rolling(SHORT).std()
        regime_features["vol_medium"] = returns.rolling(MEDIUM).std()
        regime_features["vol_long"] = returns.rolling(LONG).std()

        # Garman-Klass volatility
        log_hl = np.log(df["High"] / df["Low"]) ** 2
        log_co = np.log(df["Close"] / df["Open"]) ** 2
        gk_vol = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(MEDIUM).mean()
        )
        regime_features["gk_volatility"] = gk_vol

        # Volatility ratio (expansión/contracción)
        regime_features["vol_ratio"] = (
            regime_features["vol_short"] / regime_features["vol_long"]
        )

        # Volume ratio
        regime_features["volume_ratio"] = (
            df["Volume"] / df["Volume"].rolling(MEDIUM).mean()
        )

        # RSI manual
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(RSI_WINDOW).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_WINDOW).mean()
        rs = gain / loss
        regime_features["rsi"] = 100 - (100 / (1 + rs))

        # Distancia desde SMA estructural
        sma_structural = df["Close"].rolling(long_structural).mean()
        regime_features["dist_sma"] = (df["Close"] - sma_structural) / sma_structural

        # Drawdown desde máximo rolling
        rolling_high = df["Close"].rolling(long_structural).max()
        regime_features["drawdown"] = (df["Close"] - rolling_high) / rolling_high

        # ── 2. Limpieza ──────────────────────────────────
        regime_features = regime_features.replace([np.inf, -np.inf], np.nan)

        # Si volume_ratio es todo NaN (pares sin volumen), dropear
        if regime_features["volume_ratio"].isna().all():
            regime_features = regime_features.drop(columns=["volume_ratio"])

        regime_features = regime_features.dropna()

        if len(regime_features) < n_regimes * 5:
            import warnings
            warnings.warn(
                f"Solo {len(regime_features)} filas válidas para {n_regimes} regímenes. "
                f"Considere usar más datos (last_days más alto)."
            )

        # ── 3. Escalar y fit GMM ─────────────────────────
        scaler = StandardScaler()
        X = scaler.fit_transform(regime_features)

        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            n_init=5,
            random_state=42,
        )
        gmm.fit(X)

        # ── 4. Predecir regímenes ────────────────────────
        labels = gmm.predict(X)

        # ── 5. Mapear clusters → labels semánticos ───────
        regime_features = regime_features.copy()
        regime_features["cluster"] = labels
        mean_returns = regime_features.groupby("cluster")["returns"].mean()

        sorted_clusters = mean_returns.sort_values().index.tolist()
        # menor return → Bear (0), medio → Sideways (1), mayor → Bull (2)
        cluster_to_regime = {
            sorted_clusters[0]: 0,   # Bear
            sorted_clusters[1]: 1,   # Sideways
            sorted_clusters[2]: 2,   # Bull
        }

        # Asignar régimen mapeado a self.features (alineación por índice)
        mapped_labels = pd.Series(
            [cluster_to_regime[c] for c in labels],
            index=regime_features.index,
        )
        self.features["regime"] = np.nan
        common_idx = self.features.index.intersection(mapped_labels.index)
        self.features.loc[common_idx, "regime"] = mapped_labels.loc[common_idx].values

        # ── 6. Probabilidades del último período ─────────
        last_point = X[-1].reshape(1, -1)
        proba = gmm.predict_proba(last_point)[0]

        # Mapear probabilidades a labels semánticos
        regime_probs = {}
        for cluster_id, regime_id in cluster_to_regime.items():
            label = REGIME_LABELS[regime_id]
            regime_probs[label] = round(proba[cluster_id], 4)

        # ── 7. Guardar estado ────────────────────────────
        last_regime_id = cluster_to_regime[labels[-1]]
        regime_label = REGIME_LABELS[last_regime_id]
        self.regime = regime_label.split()[0]  # "Bull", "Bear", o "Sideways"
        self.regime_probabilities = regime_probs
        self.regime_model = gmm

        # ── 8. Print informativo ─────────────────────────
        confidence = proba[labels[-1]]
        print(f"📊 Régimen detectado: {regime_label} (confianza: {confidence:.1%})")
        print(f"   Features: {X.shape[1]} indicadores | Ventana estructural: {long_structural}d")
        print(f"   Períodos analizados: {len(regime_features)} de {n} disponibles")

        return self

    def regime_report(self) -> None:
        """
        Visualización detallada del régimen actual.

        Muestra:
        - Régimen actual con probabilidad
        - Distribución histórica de regímenes
        - Gráfico de precio coloreado por régimen
        - Métricas por régimen (return promedio, volatilidad, duración)

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado detect_regime() previamente.
        """
        self._require_regime()

        # 1. Tabla de probabilidades
        print("=" * 50)
        print("📊 REPORTE DE RÉGIMEN DE MERCADO")
        print("=" * 50)
        emoji = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}.get(self.regime, "")
        print(f"\nRégimen actual: {self.regime} {emoji}")
        print(f"\n{'Régimen':<20} {'Probabilidad':>12}")
        print("-" * 34)
        for label, prob in self.regime_probabilities.items():
            bar = "█" * int(prob * 20)
            print(f"{label:<20} {prob:>10.1%}  {bar}")

        # 2. Estadísticas por régimen
        df = self.features.dropna(subset=["regime"]).copy()
        print(f"\n{'Régimen':<20} {'Retorno Prom':>14} {'Volatilidad':>14} {'Períodos':>10} {'Duración Prom':>15}")
        print("-" * 75)

        for regime_id, label in REGIME_LABELS.items():
            mask = df["regime"] == regime_id
            subset = df[mask]
            if len(subset) == 0:
                continue

            avg_return = subset["returns"].mean()
            avg_vol = subset["volatility_20"].mean()
            n_periods = len(subset)

            # Duración promedio (rachas consecutivas)
            regime_series = mask.astype(int)
            changes = regime_series.diff().fillna(0) != 0
            groups = changes.cumsum()
            streaks = regime_series.groupby(groups).sum()
            streaks = streaks[streaks > 0]
            avg_duration = streaks.mean() if len(streaks) > 0 else 0

            print(
                f"{label:<20} {avg_return:>13.4%} {avg_vol:>13.4f} {n_periods:>10} {avg_duration:>14.1f}"
            )

        # 3. Gráfico Plotly: precio con background coloreado por régimen
        regime_colors = {
            0: COLOR_PALETTE["red"],     # Bear
            1: COLOR_PALETTE["yellow"],  # Sideways
            2: COLOR_PALETTE["green"],   # Bull
        }

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Precio",
            line=dict(color="white", width=1.5),
        ))

        # Agregar rectángulos coloreados por régimen
        regime_col = df["regime"].dropna()
        if len(regime_col) > 0:
            current_regime = regime_col.iloc[0]
            start_idx = regime_col.index[0]

            for i in range(1, len(regime_col)):
                if regime_col.iloc[i] != current_regime or i == len(regime_col) - 1:
                    end_idx = regime_col.index[i]
                    fig.add_vrect(
                        x0=start_idx,
                        x1=end_idx,
                        fillcolor=regime_colors.get(int(current_regime), COLOR_PALETTE["gray"]),
                        opacity=0.15,
                        line_width=0,
                    )
                    current_regime = regime_col.iloc[i]
                    start_idx = regime_col.index[i]

        fig.update_layout(
            title=f"Precio con Regímenes de Mercado — {getattr(self, 'symbol', '')}",
            xaxis_title="Fecha",
            yaxis_title="Precio (USDT)",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            height=500,
        )

        fig.show()

    def recommend_strategies(self) -> None:
        """
        Recomienda estrategias de trading basadas en el régimen actual.

        Para cada estrategia del registry:
        1. Ejecuta un backtest rápido en datos del régimen actual
        2. Calcula Sharpe ratio, win rate, total return
        3. Rankea estrategias de mejor a peor
        4. Indica cuáles son recomendadas (🟢) y cuáles no (🔴)

        Mapping régimen → estrategia:
        - Bull: Trend Following, Momentum > Mean Reversion
        - Bear: Mean Reversion, Short Momentum > Trend Following
        - Sideways: Mean Reversion, Range Trading > Trend Following

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado detect_regime() previamente.
        """
        self._require_regime()

        df = self.features.dropna(subset=["regime"]).copy()

        # 1. Generar señales desde OHLCV completo (self.data tiene más historia)
        ohlcv = self.data.copy()
        strategy_signals = {}

        # Trend Following — SMA Crossover
        sma_short = ohlcv["Close"].rolling(20).mean()
        sma_long = ohlcv["Close"].rolling(50).mean()
        tf_signal = pd.Series(0, index=ohlcv.index)
        tf_signal[sma_short > sma_long] = 1
        tf_signal[sma_short < sma_long] = -1
        strategy_signals["trend_following"] = tf_signal.reindex(df.index).fillna(0)

        # Mean Reversion — Bollinger Bands
        bb_mid = ohlcv["Close"].rolling(20).mean()
        bb_std = ohlcv["Close"].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        mr_signal = pd.Series(0, index=ohlcv.index)
        mr_signal[ohlcv["Close"] < bb_lower] = 1
        mr_signal[ohlcv["Close"] > bb_upper] = -1
        strategy_signals["mean_reversion"] = mr_signal.reindex(df.index).fillna(0)

        # Momentum — RSI + Volume
        delta = ohlcv["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        volume_change = ohlcv["Volume"].pct_change()
        mom_signal = pd.Series(0, index=ohlcv.index)
        mom_signal[(rsi > 50) & (volume_change > 0)] = 1
        mom_signal[(rsi < 50) & (volume_change < 0)] = -1
        strategy_signals["momentum"] = mom_signal.reindex(df.index).fillna(0)

        # 2. Filtrar a períodos del régimen actual
        regime_name_to_id = {"Bull": 2, "Bear": 0, "Sideways": 1}
        current_regime_id = regime_name_to_id[self.regime]
        regime_mask = df["regime"] == current_regime_id
        returns = df["returns"]

        # 3. Calcular métricas por estrategia
        results = []
        for key, info in STRATEGY_REGISTRY.items():
            signals = strategy_signals[key]

            # Filtrar solo períodos del régimen actual
            regime_signals = signals[regime_mask]
            regime_returns = returns[regime_mask]

            strategy_returns = regime_signals.shift(1) * regime_returns
            strategy_returns = strategy_returns.dropna()

            # Sharpe Ratio (anualizado para crypto: 365 días)
            if strategy_returns.std() != 0 and len(strategy_returns) > 0:
                sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365)
            else:
                sharpe = 0.0

            # Win Rate
            active_returns = strategy_returns[strategy_returns != 0]
            if len(active_returns) > 0:
                win_rate = (active_returns > 0).sum() / len(active_returns)
            else:
                win_rate = 0.0

            # Total Return
            total_return = (1 + strategy_returns).cumprod().iloc[-1] - 1 if len(strategy_returns) > 0 else 0.0

            # Recomendación basada en régimen
            if self.regime in info["best_regimes"]:
                recommendation = "🟢 Recomendada"
            elif self.regime in info["worst_regimes"]:
                recommendation = "🔴 No recomendada"
            else:
                recommendation = "🟡 Neutral"

            results.append({
                "key": key,
                "name": info["name"],
                "sharpe": sharpe,
                "win_rate": win_rate,
                "total_return": total_return,
                "recommendation": recommendation,
            })

        # 4. Rankear por Sharpe ratio (descendente)
        results.sort(key=lambda x: x["sharpe"], reverse=True)

        # 5. Print tabla formateada
        print("=" * 85)
        print(f"📈 ESTRATEGIAS RECOMENDADAS — Régimen: {self.regime}")
        print("=" * 85)
        print(f"\n{'#':<4} {'Estrategia':<35} {'Sharpe':>8} {'Win Rate':>10} {'Return':>10} {'Señal'}")
        print("-" * 85)

        for i, r in enumerate(results, 1):
            print(
                f"{i:<4} {r['name']:<35} {r['sharpe']:>8.2f} {r['win_rate']:>9.1%} "
                f"{r['total_return']:>9.2%}  {r['recommendation']}"
            )

    def select_strategy(self, strategy: str) -> "RegimeMixin":
        """
        Selecciona una estrategia de trading.

        Parameters
        ----------
        strategy : str
            Nombre de la estrategia. Opciones:
            "trend_following", "mean_reversion", "momentum".

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Raises
        ------
        ValueError
            Si la estrategia no está en el registry.
        """
        if strategy not in STRATEGY_REGISTRY:
            available = list(STRATEGY_REGISTRY.keys())
            raise ValueError(
                f"Estrategia '{strategy}' no válida. Opciones: {available}"
            )

        self.selected_strategy = strategy
        info = STRATEGY_REGISTRY[strategy]
        print(f"✅ Estrategia seleccionada: {info['name']}")
        print(f"   {info['description']}")
        return self
