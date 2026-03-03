"""Constantes globales del CryptoBot."""

VALID_TIMEFRAMES = ["15m", "30m", "1h", "4h", "1d"]

REGIME_LABELS = {0: "Bear 🔴", 1: "Sideways 🟡", 2: "Bull 🟢"}

STRATEGY_REGISTRY = {
    "trend_following": {
        "name": "Trend Following (SMA Crossover)",
        "description": "Sigue la dirección de la tendencia usando cruces de medias móviles",
        "best_regimes": ["Bull", "Bear"],
        "worst_regimes": ["Sideways"],
        "rationale": "Funciona mejor en mercados con tendencia (Bull/Bear) porque las medias móviles capturan movimientos sostenidos sin generar señales falsas",
    },
    "mean_reversion": {
        "name": "Mean Reversion (Bollinger Bands)",
        "description": "Apuesta a que el precio regresa a su media cuando se aleja demasiado",
        "best_regimes": ["Sideways"],
        "worst_regimes": ["Bull", "Bear"],
        "rationale": "Funciona mejor en mercados laterales (Sideways) porque el precio oscila en un rango predecible, permitiendo comprar en soporte y vender en resistencia",
    },
    "momentum": {
        "name": "Momentum (RSI + Volume)",
        "description": "Identifica movimientos fuertes y se sube a ellos",
        "best_regimes": ["Bull"],
        "worst_regimes": ["Sideways"],
        "rationale": "Funciona mejor en mercados alcistas (Bull) porque los movimientos fuertes de precio con volumen alto tienden a continuar",
    },
    "breakout": {
        "name": "Breakout Trading (Donchian + Squeeze)",
        "description": "Detecta rupturas de rangos de consolidación usando canales Donchian y compresión de Bollinger",
        "best_regimes": ["Bull", "Bear"],
        "worst_regimes": ["Sideways"],
        "rationale": "Funciona mejor en mercados con tendencia (Bull/Bear) porque las rupturas generan movimientos direccionales fuertes. En Sideways produce falsos breakouts",
    },
    "stat_arb": {
        "name": "Statistical Arbitrage (Pairs Trading)",
        "description": "Explota la reversión a la media del spread entre dos activos correlacionados",
        "best_regimes": ["Sideways", "Bear"],
        "worst_regimes": ["Bull"],
        "rationale": "Funciona mejor en Sideways/Bear porque es market-neutral y se beneficia de spreads estables. En Bull las tendencias fuertes pueden romper la cointegración",
    },
    "volatility": {
        "name": "Volatility Trading (Vol Mean Reversion)",
        "description": "Opera ciclos de expansión/contracción de volatilidad",
        "best_regimes": ["Sideways", "Bear"],
        "worst_regimes": ["Bull"],
        "rationale": "Funciona mejor en Sideways/Bear donde la volatilidad es alta y mean-reverts. En Bull la vol suele ser baja y estable",
    },
}

# ── Defaults ─────────────────────────────────────────────
DEFAULT_EXCHANGE = "binanceus"
DEFAULT_MAX_POSITION_PCT = 0.10
DEFAULT_STOP_LOSS_PCT = 0.05
DEFAULT_TAKE_PROFIT_PCT = 0.10

COLOR_PALETTE = {
    "yellow": "#F0B90B",
    "dark": "#1E2329",
    "green": "#0ECB81",
    "red": "#F6465D",
    "gray": "#474D57",
}
