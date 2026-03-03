"""Mixin para paper trading en testnet."""

from datetime import datetime


class TradingMixin:
    """Métodos de paper trading: connect_testnet(), execute(), status()."""

    def connect_testnet(self, api_key: str, api_secret: str) -> "TradingMixin":
        """
        Conecta al testnet del exchange para paper trading.

        Parameters
        ----------
        api_key : str
            API key del testnet (NO usar keys de producción).
        api_secret : str
            API secret del testnet.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Notes
        -----
        Cada exchange tiene su propio testnet:
        - Bybit: testnet.bybit.com
        - Binance: testnet.binance.vision
        - OKX: demo mode dentro de la plataforma

        ⚠️  NUNCA uses API keys de tu cuenta real aquí.
        """
        import ccxt

        exchange_class = getattr(ccxt, self.exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{self.exchange_id}' no soportado por CCXT.")

        self._exchange_testnet = exchange_class({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "sandbox": True,
        })

        # Activar modo sandbox (testnet)
        self._exchange_testnet.set_sandbox_mode(True)

        # Verificar conexión
        try:
            balance = self._exchange_testnet.fetch_balance()
            usdt_balance = balance.get("USDT", {}).get("free", 0)
            self._testnet_connected = True

            print(f"✅ Conectado al testnet de {self.exchange_id.capitalize()}")
            print(f"   Balance USDT: ${usdt_balance:,.2f}")
            print(f"   ⚠️  Esto es dinero de prueba — no es real.")

        except Exception as e:
            self._testnet_connected = False
            raise RuntimeError(
                f"❌ Error conectando al testnet: {e}\n"
                f"   Verifica que tus API keys sean del testnet, no de producción."
            )

        return self

    def execute(self, mode: str = "paper") -> dict:
        """
        Ejecuta la señal más reciente.

        Parameters
        ----------
        mode : str, default "paper"
            Modo de ejecución:
            - "paper": ejecuta en testnet (requiere connect_testnet())
            - "live": ⛔ DESHABILITADO — solo para referencia educativa

        Returns
        -------
        dict
            Información del trade ejecutado:
            {type, symbol, amount, price, stop_loss, take_profit, timestamp}

        Notes
        -----
        Risk management se aplica automáticamente:
        - Position size basado en max_position_pct
        - Stop loss basado en stop_loss_pct
        - Take profit basado en take_profit_pct

        Raises
        ------
        RuntimeError
            Si no hay señales o no está conectado al testnet.
        ValueError
            Si mode="live" (no permitido en este curso).
        """
        if mode == "live":
            raise ValueError(
                "⛔ Modo 'live' deshabilitado. Este bot es educativo. "
                "Usa mode='paper' con el testnet."
            )
        self._require_signals()
        self._require_testnet()

        # ── 1. Obtener última señal ──────────────────────
        last_signal = self.signals.iloc[-1]
        signal_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
        signal_type = signal_map[last_signal]

        if last_signal == 0:
            print("⏸️  Señal actual: HOLD — Sin acción")
            return {"type": "HOLD", "symbol": self.symbol, "timestamp": datetime.now().isoformat()}

        # ── 2. Obtener balance y calcular position size ──
        balance = self._exchange_testnet.fetch_balance()
        usdt_free = balance.get("USDT", {}).get("free", 0)

        if usdt_free <= 0 and last_signal == 1:
            print("❌ Balance USDT insuficiente para comprar")
            return {"type": "ERROR", "reason": "insufficient_balance"}

        # ── 3. Calcular montos ───────────────────────────
        ticker = self._exchange_testnet.fetch_ticker(self._pair)
        current_price = ticker["last"]

        if last_signal == 1:  # BUY
            position_value = usdt_free * self.max_position_pct
            amount = position_value / current_price
            side = "buy"
        else:  # SELL
            # Vender la cantidad que tenemos del activo
            asset_balance = balance.get(self.symbol, {}).get("free", 0)
            if asset_balance <= 0:
                print(f"❌ No hay {self.symbol} para vender")
                return {"type": "ERROR", "reason": "no_asset_to_sell"}
            amount = asset_balance
            side = "sell"

        # ── 4. Calcular SL y TP ──────────────────────────
        if side == "buy":
            stop_loss = current_price * (1 - self.stop_loss_pct)
            take_profit = current_price * (1 + self.take_profit_pct)
        else:
            stop_loss = current_price * (1 + self.stop_loss_pct)
            take_profit = current_price * (1 - self.take_profit_pct)

        # ── 5. Ejecutar orden ────────────────────────────
        try:
            order = self._exchange_testnet.create_market_order(
                self._pair, side, amount
            )

            trade = {
                "timestamp": datetime.now().isoformat(),
                "type": signal_type,
                "symbol": self.symbol,
                "pair": self._pair,
                "side": side,
                "amount": amount,
                "price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_id": order.get("id"),
                "status": "filled",
            }

            self.trades.append(trade)

            emoji = "🟢" if side == "buy" else "🔴"
            print(f"{emoji} Orden ejecutada: {side.upper()} {amount:.6f} {self.symbol}")
            print(f"   Precio: ${current_price:,.2f}")
            print(f"   Valor: ${amount * current_price:,.2f}")
            print(f"   Stop Loss: ${stop_loss:,.2f} ({self.stop_loss_pct:.0%})")
            print(f"   Take Profit: ${take_profit:,.2f} ({self.take_profit_pct:.0%})")

            return trade

        except Exception as e:
            print(f"❌ Error ejecutando orden: {e}")
            return {"type": "ERROR", "reason": str(e)}

    def status(self) -> None:
        """
        Muestra estado actual del bot en testnet.

        Incluye:
        - Balance actual (USDT + cripto)
        - Posiciones abiertas
        - Último trade ejecutado
        - P&L no realizado
        - P&L total

        Raises
        ------
        RuntimeError
            Si no está conectado al testnet.
        """
        self._require_testnet()

        balance = self._exchange_testnet.fetch_balance()

        print("=" * 60)
        print("📊 ESTADO DEL BOT — Testnet")
        print("=" * 60)

        # ── Balance ──────────────────────────────────────
        usdt_free = balance.get("USDT", {}).get("free", 0)
        usdt_used = balance.get("USDT", {}).get("used", 0)
        asset_free = balance.get(self.symbol, {}).get("free", 0)
        asset_used = balance.get(self.symbol, {}).get("used", 0)

        print(f"\n💰 Balance:")
        print(f"   USDT:   ${usdt_free:,.2f} (libre) | ${usdt_used:,.2f} (en uso)")
        print(f"   {self.symbol}:  {asset_free:.6f} (libre) | {asset_used:.6f} (en uso)")

        # ── Posición actual ──────────────────────────────
        if asset_free > 0 or asset_used > 0:
            try:
                ticker = self._exchange_testnet.fetch_ticker(self._pair)
                current_price = ticker["last"]
                position_value = (asset_free + asset_used) * current_price
                print(f"\n📈 Posición {self.symbol}:")
                print(f"   Cantidad: {asset_free + asset_used:.6f}")
                print(f"   Precio actual: ${current_price:,.2f}")
                print(f"   Valor: ${position_value:,.2f}")

                # P&L no realizado (si hay trades)
                buy_trades = [t for t in self.trades if t["type"] == "BUY"]
                if buy_trades:
                    last_buy = buy_trades[-1]
                    entry_price = last_buy["price"]
                    pnl_pct = (current_price - entry_price) / entry_price * 100
                    pnl_usd = (current_price - entry_price) * (asset_free + asset_used)
                    emoji = "🟢" if pnl_pct >= 0 else "🔴"
                    print(f"   P&L: {emoji} ${pnl_usd:,.2f} ({pnl_pct:+.2f}%)")
            except Exception:
                pass

        # ── Último trade ─────────────────────────────────
        if self.trades:
            last = self.trades[-1]
            print(f"\n📝 Último trade:")
            print(f"   {last['type']} {last['amount']:.6f} {last['symbol']} @ ${last['price']:,.2f}")
            print(f"   Fecha: {last['timestamp']}")
            print(f"   SL: ${last.get('stop_loss', 0):,.2f} | TP: ${last.get('take_profit', 0):,.2f}")
        else:
            print(f"\n📝 No hay trades registrados.")

        print("=" * 60)
