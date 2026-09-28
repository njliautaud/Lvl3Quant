#!/usr/bin/env python3
"""
rithmic_client.py — Async Rithmic WebSocket client for Linux.

Wraps the R|Protocol API (Rithmic 0.89) into a reusable class with:
  * connect / login (Ticker Plant + Order Plant, dual-socket pattern)
  * subscribe_md   (BBO + last-trade)
  * submit_order   (market or limit, BUY/SELL)
  * heartbeat loop (asyncio task)
  * structured logging to /home/jupiter/Lvl3Quant/live_trading_linux/logs/rithmic.log
  * callback registration for MD and order events

Credentials are read from environment:
    RITHMIC_SYSTEM       e.g. 'Rithmic Paper Trading' or 'Rithmic 01'
    RITHMIC_USER         AMP user id
    RITHMIC_PASSWORD     AMP password
    RITHMIC_URI          default 'wss://rituz00100.rithmic.com:443' (paper)

This is a REFACTORED version of SampleMD.py + SampleOrder.py — same wire protocol,
same protobufs, but packaged so signal_engine.py can own the event loop.

IMPORTANT:
  * Rithmic requires a separate WebSocket per infra_type (TICKER_PLANT, ORDER_PLANT).
    We open both and keep them concurrent.
  * Heartbeat cadence is driven by the server (`heartbeat_interval` returned in
    ResponseLogin).  We default to 5s; the loop self-tunes if Rithmic reports
    a different value.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import ssl
import sys
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import websockets

# ---------------------------------------------------------------------------
# Make protobuf modules importable.  The Rithmic samples dir is a flat bundle
# of `*_pb2.py` files next to SampleMD.py; we add it to sys.path.
# ---------------------------------------------------------------------------
_PB_DIR = pathlib.Path(
    os.environ.get(
        "RITHMIC_PB_DIR",
        "/home/jupiter/teleclaude-main/downloads/rithmic_api/0.89.0.0/samples/samples.py",
    )
)
if str(_PB_DIR) not in sys.path:
    sys.path.insert(0, str(_PB_DIR))

import base_pb2                                  # noqa: E402
import best_bid_offer_pb2                        # noqa: E402
import exchange_order_notification_pb2           # noqa: E402
import last_trade_pb2                            # noqa: E402
import request_account_list_pb2                  # noqa: E402
import request_heartbeat_pb2                     # noqa: E402
import request_login_info_pb2                    # noqa: E402
import request_login_pb2                         # noqa: E402
import request_logout_pb2                        # noqa: E402
import request_market_data_update_pb2            # noqa: E402
import request_new_order_pb2                     # noqa: E402
import request_subscribe_for_order_updates_pb2   # noqa: E402
import request_trade_routes_pb2                  # noqa: E402
import response_account_list_pb2                 # noqa: E402
import response_login_info_pb2                   # noqa: E402
import response_login_pb2                        # noqa: E402
import response_trade_routes_pb2                 # noqa: E402
import rithmic_order_notification_pb2            # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_DIR = pathlib.Path("/home/jupiter/Lvl3Quant/live_trading_linux/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("rithmic_client")
if not log.handlers:
    log.setLevel(logging.INFO)
    _fh = logging.FileHandler(_LOG_DIR / "rithmic.log")
    _fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    log.addHandler(_fh)
    _sh = logging.StreamHandler()
    _sh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    ))
    log.addHandler(_sh)

# ---------------------------------------------------------------------------
# Template IDs (copied from Rithmic protocol docs / samples)
# ---------------------------------------------------------------------------
TID_REQUEST_HEARTBEAT        = 18
TID_RESPONSE_HEARTBEAT       = 19
TID_REQUEST_LOGIN            = 10
TID_RESPONSE_LOGIN           = 11
TID_REQUEST_LOGOUT           = 12
TID_RESPONSE_LOGOUT          = 13
TID_REQUEST_LOGIN_INFO       = 300
TID_REQUEST_ACCOUNT_LIST     = 302
TID_REQUEST_TRADE_ROUTES     = 310
TID_REQUEST_SUBSCRIBE_OU     = 308
TID_REQUEST_NEW_ORDER        = 312
TID_RESPONSE_NEW_ORDER       = 313
TID_REQUEST_MD_UPDATE        = 100
TID_RESPONSE_MD_UPDATE       = 101
TID_LAST_TRADE               = 150
TID_BEST_BID_OFFER           = 151
TID_RITHMIC_ORDER_NOTIF      = 351
TID_EXCHANGE_ORDER_NOTIF     = 352

SSL_CERT_PATH = _PB_DIR / "rithmic_ssl_cert_auth_params"


# ---------------------------------------------------------------------------
# Normalised event types we emit to callbacks
# ---------------------------------------------------------------------------
@dataclass
class BBOEvent:
    symbol: str
    exchange: str
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    has_bid: bool
    has_ask: bool
    ssboe: int            # exchange-side seconds since epoch
    usecs: int            # microseconds component
    recv_ts: float        # local wallclock at recv


@dataclass
class TradeEvent:
    symbol: str
    exchange: str
    trade_price: float
    trade_size: int
    aggressor: int        # 1=BUY (lifts offer), 2=SELL (hits bid); 0 = unknown
    ssboe: int
    usecs: int
    recv_ts: float


@dataclass
class OrderEvent:
    notify_type: int
    status: str
    basket_id: str
    symbol: str
    exchange: str
    side: str             # 'B' or 'S' (or '' if unknown)
    quantity: int
    price: float
    fill_price: float
    fill_size: int
    text: str
    ssboe: int
    usecs: int
    recv_ts: float


MDCallback    = Callable[[object], Awaitable[None]]    # gets BBOEvent or TradeEvent
OrderCallback = Callable[[OrderEvent], Awaitable[None]]


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------
class RithmicClient:
    """Async Rithmic WebSocket client.

    Usage:
        rc = RithmicClient()                 # reads env vars
        await rc.connect()                   # logs into both plants
        rc.set_md_callback(on_md)
        rc.set_order_callback(on_order)
        await rc.subscribe_md('ESZ5', 'CME')
        # later...
        await rc.submit_order('ESZ5', 'B', 1)
        # shutdown:
        await rc.disconnect()
    """

    def __init__(
        self,
        uri:          Optional[str] = None,
        system_name:  Optional[str] = None,
        user_id:      Optional[str] = None,
        password:     Optional[str] = None,
        app_name:     str = "Lvl3Quant_LiveTrader",
        app_version:  str = "0.1.0",
        heartbeat_s:  float = 5.0,
        ssl_cert:     Optional[pathlib.Path] = None,
    ) -> None:
        self.uri         = uri         or os.environ.get("RITHMIC_URI", "wss://rituz00100.rithmic.com:443")
        self.system_name = system_name or os.environ.get("RITHMIC_SYSTEM", "")
        self.user_id     = user_id     or os.environ.get("RITHMIC_USER", "")
        self.password    = password    or os.environ.get("RITHMIC_PASSWORD", "")
        self.app_name    = app_name
        self.app_version = app_version
        self.heartbeat_s = heartbeat_s
        self.ssl_cert    = ssl_cert or SSL_CERT_PATH

        # Two WebSocket connections — one per infra type.
        self._ws_md:    Optional[websockets.WebSocketClientProtocol] = None
        self._ws_order: Optional[websockets.WebSocketClientProtocol] = None

        # Account / trade-route info resolved at login
        self.fcm_id       = ""
        self.ib_id        = ""
        self.account_id   = ""
        self.trade_route  = ""

        # Callbacks
        self._md_cb:    Optional[MDCallback]    = None
        self._order_cb: Optional[OrderCallback] = None

        # Background tasks
        self._tasks: list[asyncio.Task] = []
        self._stop  = asyncio.Event()

        # Track subscriptions for clean unsubscribe
        self._subs: list[tuple[str, str]] = []  # (exchange, symbol)

        self._connected = False

    # ------------------------------------------------------------------ SSL
    def _ssl_context(self) -> Optional[ssl.SSLContext]:
        if "wss://" not in self.uri:
            return None
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        if not self.ssl_cert.exists():
            raise FileNotFoundError(
                f"Rithmic SSL cert file not found: {self.ssl_cert}. "
                f"Set RITHMIC_PB_DIR or pass ssl_cert= explicitly."
            )
        ctx.load_verify_locations(str(self.ssl_cert))
        return ctx

    # ------------------------------------------------------------------- connect
    async def connect(self, md_only: bool = False) -> None:
        """Open WebSockets and log in.

        Args:
            md_only: If True, only open TICKER_PLANT (no ORDER_PLANT).
                     Use this for data recording where orders are not needed.
                     Reduces connection count — Rithmic limits simultaneous logins.
        """
        if not (self.system_name and self.user_id and self.password):
            raise RuntimeError(
                "Missing Rithmic credentials. "
                "Set RITHMIC_SYSTEM / RITHMIC_USER / RITHMIC_PASSWORD env vars."
            )

        ctx = self._ssl_context()
        log.info("Connecting MD socket to %s ...", self.uri)
        self._ws_md = await websockets.connect(self.uri, ssl=ctx, ping_interval=None)

        # Login to ticker plant
        await self._login(
            self._ws_md,
            request_login_pb2.RequestLogin.SysInfraType.TICKER_PLANT,
            label="MD",
        )

        if not md_only:
            log.info("Connecting ORDER socket to %s ...", self.uri)
            self._ws_order = await websockets.connect(self.uri, ssl=ctx, ping_interval=None)
            await self._login(
                self._ws_order,
                request_login_pb2.RequestLogin.SysInfraType.ORDER_PLANT,
                label="ORDER",
            )

            # Resolve account + trade route (order plant only)
            await self._fetch_login_info()

            # Subscribe to order updates
            if self.fcm_id and self.ib_id and self.account_id:
                await self._subscribe_order_updates()

        # Start background consumers + heartbeat
        self._tasks.append(asyncio.create_task(self._consume_md(), name="md_consumer"))
        if not md_only and self._ws_order is not None:
            self._tasks.append(asyncio.create_task(self._consume_order(), name="order_consumer"))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="heartbeat"))

        self._connected = True
        log.info("Rithmic connected. account=%s trade_route=%s", self.account_id, self.trade_route)

    # ------------------------------------------------------------------- login
    async def _login(
        self,
        ws: websockets.WebSocketClientProtocol,
        infra_type: int,
        label: str,
    ) -> None:
        rq = request_login_pb2.RequestLogin()
        rq.template_id      = TID_REQUEST_LOGIN
        rq.template_version = "3.9"
        rq.user_msg.append("hello")
        rq.user             = self.user_id
        rq.password         = self.password
        rq.app_name         = self.app_name
        rq.app_version      = self.app_version
        rq.system_name      = self.system_name
        rq.infra_type       = infra_type
        await ws.send(rq.SerializeToString())

        rp_buf = await ws.recv()
        rp = response_login_pb2.ResponseLogin()
        rp.ParseFromString(rp_buf)
        if rp.rp_code and rp.rp_code[0] != "0":
            raise RuntimeError(f"{label} login failed: rp_code={list(rp.rp_code)} msg={list(rp.user_msg)}")
        if rp.heartbeat_interval and rp.heartbeat_interval > 0:
            # Server-advised cadence; send at 1/2 of it to stay safe
            self.heartbeat_s = max(1.0, rp.heartbeat_interval / 2.0)
        log.info("%s login OK — hb=%ss fcm=%s ib=%s",
                 label, self.heartbeat_s, rp.fcm_id, rp.ib_id)

    # ------------------------------------------------------------- login_info
    async def _fetch_login_info(self) -> None:
        ws = self._ws_order
        assert ws is not None
        rq = request_login_info_pb2.RequestLoginInfo()
        rq.template_id = TID_REQUEST_LOGIN_INFO
        rq.user_msg.append("hello")
        await ws.send(rq.SerializeToString())

        rp_buf = await ws.recv()
        rp = response_login_info_pb2.ResponseLoginInfo()
        rp.ParseFromString(rp_buf)
        if not rp.rp_code or rp.rp_code[0] != "0":
            log.warning("ResponseLoginInfo rp_code=%s — account/route resolution may fail",
                        list(rp.rp_code))
            return

        # Account list
        rq2 = request_account_list_pb2.RequestAccountList()
        rq2.template_id = TID_REQUEST_ACCOUNT_LIST
        rq2.user_msg.append("hello")
        rq2.fcm_id    = rp.fcm_id
        rq2.ib_id     = rp.ib_id
        rq2.user_type = rp.user_type
        await ws.send(rq2.SerializeToString())
        while True:
            buf = await ws.recv()
            rp2 = response_account_list_pb2.ResponseAccountList()
            rp2.ParseFromString(buf)
            if (rp2.rq_handler_rp_code and rp2.rq_handler_rp_code[0] == "0"
                    and rp2.fcm_id and rp2.ib_id and rp2.account_id
                    and not self.account_id):
                self.fcm_id     = rp2.fcm_id
                self.ib_id      = rp2.ib_id
                self.account_id = rp2.account_id
                log.info("Resolved account: fcm=%s ib=%s acct=%s",
                         self.fcm_id, self.ib_id, self.account_id)
            if rp2.rp_code:
                break

        # Trade routes
        rq3 = request_trade_routes_pb2.RequestTradeRoutes()
        rq3.template_id = TID_REQUEST_TRADE_ROUTES
        rq3.user_msg.append("hello")
        rq3.subscribe_for_updates = False
        await ws.send(rq3.SerializeToString())
        while True:
            buf = await ws.recv()
            rp3 = response_trade_routes_pb2.ResponseTradeRoutes()
            rp3.ParseFromString(buf)
            # Grab the first applicable route for our fcm+ib; caller can override
            # per-symbol via set_trade_route() if needed.
            if (rp3.rq_handler_rp_code and rp3.rq_handler_rp_code[0] == "0"
                    and rp3.fcm_id == self.fcm_id and rp3.ib_id == self.ib_id
                    and not self.trade_route):
                self.trade_route = rp3.trade_route
                log.info("Resolved trade_route=%s (exchange=%s)",
                         self.trade_route, rp3.exchange)
            if rp3.rp_code:
                break

    # --------------------------------------------------------- subscribe_order
    async def _subscribe_order_updates(self) -> None:
        ws = self._ws_order
        assert ws is not None
        rq = request_subscribe_for_order_updates_pb2.RequestSubscribeForOrderUpdates()
        rq.template_id = TID_REQUEST_SUBSCRIBE_OU
        rq.user_msg.append("hello")
        rq.fcm_id     = self.fcm_id
        rq.ib_id      = self.ib_id
        rq.account_id = self.account_id
        await ws.send(rq.SerializeToString())
        log.info("Subscribed to order updates for account=%s", self.account_id)

    # -------------------------------------------------------------- callbacks
    def set_md_callback(self, cb: MDCallback) -> None:
        """Register an async callback for market-data events (BBOEvent | TradeEvent)."""
        self._md_cb = cb

    def set_order_callback(self, cb: OrderCallback) -> None:
        """Register an async callback for order events (OrderEvent)."""
        self._order_cb = cb

    # ----------------------------------------------------------- subscribe_md
    async def subscribe_md(self, symbol: str, exchange: str) -> None:
        ws = self._ws_md
        assert ws is not None, "connect() first"
        rq = request_market_data_update_pb2.RequestMarketDataUpdate()
        rq.template_id = TID_REQUEST_MD_UPDATE
        rq.user_msg.append("subscribe")
        rq.symbol   = symbol
        rq.exchange = exchange
        rq.request  = request_market_data_update_pb2.RequestMarketDataUpdate.Request.SUBSCRIBE
        rq.update_bits = (
            request_market_data_update_pb2.RequestMarketDataUpdate.UpdateBits.LAST_TRADE
            | request_market_data_update_pb2.RequestMarketDataUpdate.UpdateBits.BBO
        )
        await ws.send(rq.SerializeToString())
        self._subs.append((exchange, symbol))
        log.info("Subscribed MD: %s @ %s", symbol, exchange)

    async def unsubscribe_md(self, symbol: str, exchange: str) -> None:
        ws = self._ws_md
        if ws is None:
            return
        rq = request_market_data_update_pb2.RequestMarketDataUpdate()
        rq.template_id = TID_REQUEST_MD_UPDATE
        rq.user_msg.append("unsubscribe")
        rq.symbol   = symbol
        rq.exchange = exchange
        rq.request  = request_market_data_update_pb2.RequestMarketDataUpdate.Request.UNSUBSCRIBE
        rq.update_bits = (
            request_market_data_update_pb2.RequestMarketDataUpdate.UpdateBits.LAST_TRADE
            | request_market_data_update_pb2.RequestMarketDataUpdate.UpdateBits.BBO
        )
        try:
            await ws.send(rq.SerializeToString())
        except Exception as e:
            log.warning("unsubscribe %s@%s failed: %s", symbol, exchange, e)

    # -------------------------------------------------------------- submit_order
    async def submit_order(
        self,
        symbol:      str,
        side:        str,                  # 'B' or 'S'
        qty:         int,
        price_type:  str = "MARKET",       # 'MARKET' or 'LIMIT'
        limit_price: Optional[float] = None,
        exchange:    Optional[str] = None,
        trade_route: Optional[str] = None,
        duration:    str = "DAY",          # 'DAY' | 'GTC' | 'IOC' | 'FOK'
    ) -> None:
        """Submit a new order via the order plant.

        Orders are fire-and-forget — status comes back via the order callback.
        """
        ws = self._ws_order
        if ws is None:
            raise RuntimeError("Not connected. Call connect() first.")
        if not (self.fcm_id and self.ib_id and self.account_id):
            raise RuntimeError("Account not resolved — cannot submit order.")

        # Fall back to the default trade_route resolved at login if caller didn't
        # pass one.  For live trading on a new exchange you should look up the
        # appropriate route explicitly.
        tr = trade_route or self.trade_route
        if not tr:
            raise RuntimeError("No trade_route available — cannot submit order.")

        rq = request_new_order_pb2.RequestNewOrder()
        rq.template_id = TID_REQUEST_NEW_ORDER
        rq.user_msg.append("hello")
        rq.fcm_id     = self.fcm_id
        rq.ib_id      = self.ib_id
        rq.account_id = self.account_id
        rq.exchange   = exchange or (self._subs[-1][0] if self._subs else "")
        rq.symbol     = symbol
        rq.quantity   = int(qty)
        rq.trade_route = tr

        side_u = side.upper()
        if side_u in ("B", "BUY"):
            rq.transaction_type = request_new_order_pb2.RequestNewOrder.TransactionType.BUY
        elif side_u in ("S", "SELL"):
            rq.transaction_type = request_new_order_pb2.RequestNewOrder.TransactionType.SELL
        else:
            raise ValueError(f"Bad side: {side!r}")

        dur_u = duration.upper()
        dur_map = {
            "DAY": request_new_order_pb2.RequestNewOrder.Duration.DAY,
            "GTC": request_new_order_pb2.RequestNewOrder.Duration.GTC,
            "IOC": request_new_order_pb2.RequestNewOrder.Duration.IOC,
            "FOK": request_new_order_pb2.RequestNewOrder.Duration.FOK,
        }
        if dur_u not in dur_map:
            raise ValueError(f"Bad duration: {duration!r}")
        rq.duration = dur_map[dur_u]

        pt_u = price_type.upper()
        if pt_u == "MARKET":
            rq.price_type = request_new_order_pb2.RequestNewOrder.PriceType.MARKET
        elif pt_u == "LIMIT":
            if limit_price is None:
                raise ValueError("limit_price required for LIMIT order")
            rq.price_type = request_new_order_pb2.RequestNewOrder.PriceType.LIMIT
            rq.price = float(limit_price)
        else:
            raise ValueError(f"Unsupported price_type: {price_type!r}")

        rq.manual_or_auto = request_new_order_pb2.RequestNewOrder.OrderPlacement.AUTO

        try:
            await ws.send(rq.SerializeToString())
            log.info("Sent order: %s %s %d %s @ %s (route=%s)",
                     side_u, symbol, qty, pt_u,
                     limit_price if limit_price is not None else "mkt",
                     tr)
        except Exception as e:
            log.error("submit_order send failed: %s", e)
            raise

    # -------------------------------------------------------------- heartbeat
    async def _send_heartbeat(self, ws: websockets.WebSocketClientProtocol) -> None:
        rq = request_heartbeat_pb2.RequestHeartbeat()
        rq.template_id = TID_REQUEST_HEARTBEAT
        try:
            await ws.send(rq.SerializeToString())
        except Exception as e:
            log.warning("heartbeat send failed: %s", e)

    @staticmethod
    def _is_open(ws) -> bool:
        """Check if websocket is open — compatible with websockets 10-16+."""
        try:
            if hasattr(ws, 'open'):
                return ws.open
            # websockets >= 14: .protocol.state or just check not closed
            from websockets import State
            if hasattr(ws, 'protocol') and hasattr(ws.protocol, 'state'):
                return ws.protocol.state == State.OPEN
            # Fallback: assume open if not explicitly closed
            return True
        except Exception:
            return False

    async def _heartbeat_loop(self) -> None:
        try:
            while not self._stop.is_set():
                await asyncio.sleep(self.heartbeat_s)
                if self._ws_md is not None and self._is_open(self._ws_md):
                    await self._send_heartbeat(self._ws_md)
                if self._ws_order is not None and self._is_open(self._ws_order):
                    await self._send_heartbeat(self._ws_order)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("heartbeat loop crashed: %s", e)

    # -------------------------------------------------------------- consumers
    async def _consume_md(self) -> None:
        ws = self._ws_md
        assert ws is not None
        try:
            while not self._stop.is_set():
                try:
                    buf = await asyncio.wait_for(ws.recv(), timeout=self.heartbeat_s * 2)
                except asyncio.TimeoutError:
                    continue  # heartbeat loop will have pinged
                except websockets.ConnectionClosed as e:
                    log.warning("MD socket closed: %s", e)
                    return

                await self._dispatch_md(buf)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("MD consumer crashed: %s", e)

    async def _consume_order(self) -> None:
        ws = self._ws_order
        assert ws is not None
        try:
            while not self._stop.is_set():
                try:
                    buf = await asyncio.wait_for(ws.recv(), timeout=self.heartbeat_s * 2)
                except asyncio.TimeoutError:
                    continue
                except websockets.ConnectionClosed as e:
                    log.warning("ORDER socket closed: %s", e)
                    return

                await self._dispatch_order(buf)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.exception("ORDER consumer crashed: %s", e)

    # -------------------------------------------------------------- dispatch
    async def _dispatch_md(self, buf: bytes) -> None:
        base = base_pb2.Base()
        try:
            base.ParseFromString(buf)
        except Exception as e:
            log.warning("MD parse base failed: %s", e)
            return
        tid = base.template_id
        now = time.time()

        if tid == TID_BEST_BID_OFFER:
            m = best_bid_offer_pb2.BestBidOffer()
            m.ParseFromString(buf)
            has_bid = bool(m.presence_bits & best_bid_offer_pb2.BestBidOffer.PresenceBits.BID)
            has_ask = bool(m.presence_bits & best_bid_offer_pb2.BestBidOffer.PresenceBits.ASK)
            ev = BBOEvent(
                symbol=m.symbol, exchange=m.exchange,
                bid_price=float(m.bid_price) if has_bid else 0.0,
                bid_size=int(m.bid_size)   if has_bid else 0,
                ask_price=float(m.ask_price) if has_ask else 0.0,
                ask_size=int(m.ask_size)   if has_ask else 0,
                has_bid=has_bid, has_ask=has_ask,
                ssboe=int(m.ssboe), usecs=int(m.usecs),
                recv_ts=now,
            )
            if self._md_cb:
                try:
                    await self._md_cb(ev)
                except Exception as e:
                    log.exception("md_cb failed for BBO: %s", e)

        elif tid == TID_LAST_TRADE:
            m = last_trade_pb2.LastTrade()
            m.ParseFromString(buf)
            # aggressor: BUY=1, SELL=2 in the enum; 0 if unset
            ev = TradeEvent(
                symbol=m.symbol, exchange=m.exchange,
                trade_price=float(m.trade_price),
                trade_size=int(m.trade_size),
                aggressor=int(m.aggressor),
                ssboe=int(m.ssboe), usecs=int(m.usecs),
                recv_ts=now,
            )
            if self._md_cb:
                try:
                    await self._md_cb(ev)
                except Exception as e:
                    log.exception("md_cb failed for trade: %s", e)

        elif tid in (TID_RESPONSE_MD_UPDATE, TID_RESPONSE_HEARTBEAT,
                     TID_RESPONSE_LOGOUT):
            return  # quiet
        else:
            log.debug("MD: unhandled template_id=%d", tid)

    async def _dispatch_order(self, buf: bytes) -> None:
        base = base_pb2.Base()
        try:
            base.ParseFromString(buf)
        except Exception as e:
            log.warning("ORDER parse base failed: %s", e)
            return
        tid = base.template_id
        now = time.time()

        if tid == TID_RITHMIC_ORDER_NOTIF:
            m = rithmic_order_notification_pb2.RithmicOrderNotification()
            m.ParseFromString(buf)
            side = ""
            try:
                tt_buy  = rithmic_order_notification_pb2.RithmicOrderNotification.TransactionType.BUY
                tt_sell = rithmic_order_notification_pb2.RithmicOrderNotification.TransactionType.SELL
                if m.transaction_type == tt_buy:
                    side = "B"
                elif m.transaction_type == tt_sell:
                    side = "S"
            except Exception:
                pass
            ev = OrderEvent(
                notify_type=int(m.notify_type),
                status=str(m.status),
                basket_id=str(m.basket_id),
                symbol=str(m.symbol),
                exchange=str(m.exchange),
                side=side,
                quantity=int(m.quantity),
                price=float(m.price),
                fill_price=0.0,   # rithmic-notification is status-level; fills come via exchange notif
                fill_size=0,
                text=str(m.text),
                ssboe=int(m.ssboe), usecs=int(m.usecs),
                recv_ts=now,
            )
            if self._order_cb:
                try:
                    await self._order_cb(ev)
                except Exception as e:
                    log.exception("order_cb (rithmic) failed: %s", e)
            log.info("RithmicOrderNotif type=%d status=%s basket=%s",
                     ev.notify_type, ev.status, ev.basket_id)

        elif tid == TID_EXCHANGE_ORDER_NOTIF:
            m = exchange_order_notification_pb2.ExchangeOrderNotification()
            m.ParseFromString(buf)
            side = ""
            try:
                tt_buy  = exchange_order_notification_pb2.ExchangeOrderNotification.TransactionType.BUY
                tt_sell = exchange_order_notification_pb2.ExchangeOrderNotification.TransactionType.SELL
                if m.transaction_type == tt_buy:
                    side = "B"
                elif m.transaction_type == tt_sell:
                    side = "S"
            except Exception:
                pass
            ev = OrderEvent(
                notify_type=int(m.notify_type),
                status=str(m.status),
                basket_id=str(m.basket_id),
                symbol=str(m.symbol),
                exchange=str(m.exchange),
                side=side,
                quantity=int(m.quantity),
                price=float(m.price),
                fill_price=float(m.fill_price),
                fill_size=int(m.fill_size),
                text=str(m.text),
                ssboe=int(m.ssboe), usecs=int(m.usecs),
                recv_ts=now,
            )
            if self._order_cb:
                try:
                    await self._order_cb(ev)
                except Exception as e:
                    log.exception("order_cb (exchange) failed: %s", e)
            log.info("ExchOrderNotif type=%d status=%s fill=%s@%s",
                     ev.notify_type, ev.status, ev.fill_size, ev.fill_price)

        elif tid in (TID_RESPONSE_NEW_ORDER, TID_RESPONSE_HEARTBEAT,
                     TID_RESPONSE_LOGOUT):
            return
        else:
            log.debug("ORDER: unhandled template_id=%d", tid)

    # -------------------------------------------------------------- disconnect
    async def disconnect(self) -> None:
        self._stop.set()

        # Unsubscribe MD
        for exch, sym in list(self._subs):
            try:
                await self.unsubscribe_md(sym, exch)
            except Exception:
                pass
        self._subs.clear()

        # Logout on both sockets
        for ws, label in ((self._ws_md, "MD"), (self._ws_order, "ORDER")):
            if ws is None:
                continue
            try:
                rq = request_logout_pb2.RequestLogout()
                rq.template_id = TID_REQUEST_LOGOUT
                rq.user_msg.append("bye")
                await ws.send(rq.SerializeToString())
            except Exception as e:
                log.warning("%s logout failed: %s", label, e)
            try:
                await ws.close(1000, "client shutdown")
            except Exception as e:
                log.warning("%s close failed: %s", label, e)

        # Cancel tasks
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()

        self._ws_md = None
        self._ws_order = None
        self._connected = False
        log.info("RithmicClient disconnected.")
