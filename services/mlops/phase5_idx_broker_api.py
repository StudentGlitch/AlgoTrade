"""
Phase 5: IDX Broker API Integration
-------------------------------------
Localized data connector and Level 2 Order Book wrapper for the Indonesian
Stock Exchange (IDX).  This module provides:

  IDXOrderBookLevel2
    Thread-safe in-memory snapshot of the Level 2 LOB:
    bids/asks as [(price, volume)] lists, derived LOB imbalance, and
    convenience accessors for mid-price and best bid/ask.

  IDXBrokerAPIWrapper (abstract base)
    Common interface for all IDX broker backends:
    subscribe / place_order / cancel_order / get_orderbook / disconnect.

  IPOTAPIWrapper
    Placeholder for the IPOT (Indo Premier Online Technology) REST+WebSocket
    API.  Fill in real credentials and endpoint paths when you obtain access.

  MandiriSekuritasAPIWrapper
    Placeholder for Mandiri Sekuritas MOST (Mobile Online System Trading).
    Follows the same abstract interface.

  IDXDataFeedWebSocket
    Generic Level 2 LOB ingestion over a WebSocket feed (e.g. IDX datafeed,
    Mirae Asset / NextG, or any vendor publishing OHLCV + LOB in JSON).
    The connector is non-blocking: it runs in a daemon thread and exposes a
    thread-safe snapshot via get_latest_orderbook().

All classes degrade gracefully when broker credentials are absent or the
websocket-client package is unavailable — they return empty/neutral data
rather than raising at import time.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import websocket  # websocket-client package
except ImportError:
    websocket = None

try:
    import requests as _requests
except ImportError:
    _requests = None

_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Level 2 Order Book snapshot
# ---------------------------------------------------------------------------

@dataclass
class IDXOrderBookLevel2:
    """
    Immutable snapshot of the Level 2 Order Book for a single IDX instrument.

    Attributes
    ----------
    symbol      : IDX ticker (e.g. 'BBRI')
    timestamp   : UTC timestamp of the snapshot
    bids        : List of (price, volume) tuples, best bid first
    asks        : List of (price, volume) tuples, best ask first
    last_price  : Last traded price
    session     : Market session identifier ('pre_open', 'open', 'pre_close', 'post')
    """

    symbol: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bids: List[Tuple[float, int]] = field(default_factory=list)
    asks: List[Tuple[float, int]] = field(default_factory=list)
    last_price: float = 0.0
    session: str = "open"

    @property
    def best_bid(self) -> float:
        return float(self.bids[0][0]) if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return float(self.asks[0][0]) if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) / 2.0
        return self.last_price

    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.best_ask - self.best_bid
        return 0.0

    @property
    def lob_imbalance(self) -> float:
        """
        (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        in range [-1, +1].  Positive = bid-heavy (demand > supply).
        """
        bid_vol = sum(v for _, v in self.bids)
        ask_vol = sum(v for _, v in self.asks)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def is_empty(self) -> bool:
        """True when the bid side is completely empty (ARB/ARA halt condition)."""
        return len(self.bids) == 0


# ---------------------------------------------------------------------------
# Abstract broker base class
# ---------------------------------------------------------------------------

class IDXBrokerAPIWrapper(ABC):
    """
    Abstract interface for IDX broker REST/WebSocket backends.

    All concrete implementations must handle missing/invalid credentials
    gracefully and not raise at construction time.
    """

    @abstractmethod
    def connect(self) -> bool:
        """Open the connection to the broker backend. Returns True on success."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection and release resources."""

    @abstractmethod
    def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to real-time Level 2 data for a list of symbols."""

    @abstractmethod
    def get_orderbook(self, symbol: str) -> IDXOrderBookLevel2:
        """Return the latest Level 2 snapshot for a symbol."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "limit",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Submit an order.

        Args:
            symbol    : IDX ticker (e.g. 'BBRI')
            side      : 'buy' or 'sell'
            qty       : Lot quantity (1 lot = 100 shares on IDX)
            order_type: 'limit' or 'market'
            price     : Limit price in Rupiah (required for limit orders)

        Returns:
            Dict with at least keys: order_id, status, message.
        """

    @abstractmethod
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order by its broker-assigned ID."""


# ---------------------------------------------------------------------------
# IPOT (Indo Premier Online Technology) placeholder
# ---------------------------------------------------------------------------

class IPOTAPIWrapper(IDXBrokerAPIWrapper):
    """
    Placeholder connector for the IPOT (Indo Premier) broker API.

    Environment variables (set in .env):
      IPOT_API_URL      Base REST URL (e.g. https://api.indopremier.com/v1)
      IPOT_API_KEY      Client API key
      IPOT_API_SECRET   Client API secret

    Replace the stub implementations with actual IPOT REST/WebSocket calls
    once you have a licensed API subscription.
    """

    def __init__(self):
        self._base_url = os.getenv("IPOT_API_URL", "").strip()
        self._api_key = os.getenv("IPOT_API_KEY", "").strip()
        self._api_secret = os.getenv("IPOT_API_SECRET", "").strip()
        self._session = None
        self._snapshots: Dict[str, IDXOrderBookLevel2] = {}

    def enabled(self) -> bool:
        return bool(self._base_url and self._api_key)

    def connect(self) -> bool:
        if not self.enabled():
            _LOG.warning("IPOT API credentials not configured; running in placeholder mode.")
            return False
        _LOG.info("IPOT: connecting to %s", self._base_url)
        # TODO: implement OAuth/session handshake when credentials are present
        return False

    def disconnect(self) -> None:
        _LOG.info("IPOT: disconnected (placeholder).")

    def subscribe(self, symbols: List[str]) -> None:
        _LOG.info("IPOT: subscribe stub called for %s", symbols)

    def get_orderbook(self, symbol: str) -> IDXOrderBookLevel2:
        return self._snapshots.get(symbol, IDXOrderBookLevel2(symbol=symbol))

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "limit",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        _LOG.info("IPOT: place_order stub → symbol=%s side=%s qty=%d", symbol, side, qty)
        return {"order_id": "", "status": "placeholder", "message": "IPOT API not configured"}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        _LOG.info("IPOT: cancel_order stub → order_id=%s", order_id)
        return {"order_id": order_id, "status": "placeholder", "message": "IPOT API not configured"}


# ---------------------------------------------------------------------------
# Mandiri Sekuritas MOST placeholder
# ---------------------------------------------------------------------------

class MandiriSekuritasAPIWrapper(IDXBrokerAPIWrapper):
    """
    Placeholder connector for Mandiri Sekuritas MOST (Mobile Online System Trading).

    Environment variables:
      MANDIRI_API_URL     Base REST URL
      MANDIRI_API_TOKEN   Bearer token

    Replace stubs with real MOST API calls when credentials are available.
    """

    def __init__(self):
        self._base_url = os.getenv("MANDIRI_API_URL", "").strip()
        self._token = os.getenv("MANDIRI_API_TOKEN", "").strip()
        self._snapshots: Dict[str, IDXOrderBookLevel2] = {}

    def enabled(self) -> bool:
        return bool(self._base_url and self._token)

    def connect(self) -> bool:
        if not self.enabled():
            _LOG.warning("Mandiri API credentials not configured; running in placeholder mode.")
            return False
        return False

    def disconnect(self) -> None:
        _LOG.info("Mandiri Sekuritas: disconnected (placeholder).")

    def subscribe(self, symbols: List[str]) -> None:
        _LOG.info("Mandiri: subscribe stub for %s", symbols)

    def get_orderbook(self, symbol: str) -> IDXOrderBookLevel2:
        return self._snapshots.get(symbol, IDXOrderBookLevel2(symbol=symbol))

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: int,
        order_type: str = "limit",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {"order_id": "", "status": "placeholder", "message": "Mandiri API not configured"}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        return {"order_id": order_id, "status": "placeholder", "message": "Mandiri API not configured"}


# ---------------------------------------------------------------------------
# Generic IDX WebSocket Level 2 data feed
# ---------------------------------------------------------------------------

@dataclass
class IDXDataFeedConfig:
    """Configuration for the IDX WebSocket Level 2 data feed."""

    ws_url: str = ""
    """WebSocket endpoint URL (override via IDX_WS_URL env var)."""

    auth_token: str = ""
    """Bearer / API-key token (override via IDX_WS_TOKEN env var)."""

    reconnect_delay: float = 5.0
    """Seconds to wait before attempting reconnection."""

    max_queue_size: int = 1000
    """Maximum number of raw messages to buffer in the internal queue."""


class IDXDataFeedWebSocket:
    """
    Non-blocking Level 2 IDX data feed over a vendor WebSocket.

    Supported message formats (auto-detected):
      { "type": "lob", "symbol": "BBRI", "bids": [[price, vol], ...],
        "asks": [[price, vol], ...], "last": 1500, "session": "open" }

      { "type": "trade", "symbol": "BBRI", "price": 1505, "volume": 200 }

    Unknown message types are silently ignored.

    Thread safety: all snapshot state is protected by a threading.Lock.
    """

    def __init__(self, cfg: Optional[IDXDataFeedConfig] = None):
        self._cfg = cfg or IDXDataFeedConfig(
            ws_url=os.getenv("IDX_WS_URL", ""),
            auth_token=os.getenv("IDX_WS_TOKEN", ""),
        )
        self._snapshots: Dict[str, IDXOrderBookLevel2] = {}
        self._lock = threading.Lock()
        self._raw_queue: queue.Queue = queue.Queue(maxsize=self._cfg.max_queue_size)
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._subscribed: List[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enabled(self) -> bool:
        return bool(self._cfg.ws_url)

    def connect(self, symbols: Optional[List[str]] = None) -> bool:
        """
        Open WebSocket connection and subscribe to LOB updates.

        Args:
            symbols: Optional list of IDX tickers to subscribe. Can also be
                     set later via subscribe().

        Returns:
            True if the connection thread was started; False if not configured.
        """
        if not self.enabled():
            _LOG.warning("IDXDataFeedWebSocket: IDX_WS_URL not set; running in no-op mode.")
            return False
        if websocket is None:
            _LOG.warning("IDXDataFeedWebSocket: websocket-client not installed.")
            return False
        if symbols:
            self._subscribed = list(symbols)
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def subscribe(self, symbols: List[str]) -> None:
        """Add symbols to the active subscription list."""
        self._subscribed = list(set(self._subscribed + symbols))
        if self._ws is not None:
            try:
                self._ws.send(json.dumps({"action": "subscribe", "symbols": symbols}))
            except Exception as exc:  # pylint: disable=broad-except
                _LOG.warning("IDXDataFeedWebSocket.subscribe error: %s", exc)

    def get_latest_orderbook(self, symbol: str) -> IDXOrderBookLevel2:
        """Thread-safe access to the latest LOB snapshot for a symbol."""
        with self._lock:
            return self._snapshots.get(symbol, IDXOrderBookLevel2(symbol=symbol))

    def get_lob_imbalance(self, symbol: str) -> float:
        """Convenience: return current LOB imbalance for a symbol."""
        return self.get_latest_orderbook(symbol).lob_imbalance

    def disconnect(self) -> None:
        """Stop the feed thread and close the WebSocket."""
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # pylint: disable=broad-except
                pass
        if self._thread is not None:
            self._thread.join(timeout=10.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal WebSocket callbacks
    # ------------------------------------------------------------------

    def _on_open(self, ws) -> None:
        _LOG.info("IDXDataFeedWebSocket: connected to %s", self._cfg.ws_url)
        if self._cfg.auth_token:
            ws.send(json.dumps({"action": "auth", "token": self._cfg.auth_token}))
        if self._subscribed:
            ws.send(json.dumps({"action": "subscribe", "symbols": self._subscribed}))

    def _on_message(self, _ws, raw: str) -> None:
        try:
            self._raw_queue.put_nowait(raw)
        except queue.Full:
            pass
        self._process_message(raw)

    def _on_error(self, _ws, error) -> None:
        _LOG.warning("IDXDataFeedWebSocket error: %s", error)

    def _on_close(self, _ws, code, msg) -> None:
        _LOG.info("IDXDataFeedWebSocket closed (code=%s): %s", code, msg)

    def _process_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        msg_type = payload.get("type", "")
        symbol = payload.get("symbol", "")
        if not symbol:
            return
        ts = datetime.now(timezone.utc)

        if msg_type == "lob":
            bids = [(float(p), int(v)) for p, v in payload.get("bids", [])]
            asks = [(float(p), int(v)) for p, v in payload.get("asks", [])]
            snap = IDXOrderBookLevel2(
                symbol=symbol,
                timestamp=ts,
                bids=sorted(bids, reverse=True),
                asks=sorted(asks),
                last_price=float(payload.get("last", 0.0)),
                session=payload.get("session", "open"),
            )
            with self._lock:
                self._snapshots[symbol] = snap

        elif msg_type == "trade":
            with self._lock:
                existing = self._snapshots.get(symbol)
                if existing is None:
                    existing = IDXOrderBookLevel2(symbol=symbol, timestamp=ts)
                updated = IDXOrderBookLevel2(
                    symbol=existing.symbol,
                    timestamp=ts,
                    bids=existing.bids,
                    asks=existing.asks,
                    last_price=float(payload.get("price", existing.last_price)),
                    session=existing.session,
                )
                self._snapshots[symbol] = updated

    def _run_loop(self) -> None:
        """Reconnecting WebSocket run loop executed in a daemon thread."""
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self._cfg.ws_url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as exc:  # pylint: disable=broad-except
                _LOG.warning("IDXDataFeedWebSocket loop error: %s", exc)
            if self._running:
                time.sleep(self._cfg.reconnect_delay)
