"""
Algorithmic Trading Execution Engine
===================================
Event-driven Backtrader engine with:
1) Custom feed for FinBERT / LSTM / LPA columns
2) Regime-switching state machine from LPA profile IDs
3) Vol-targeted stop loss + fractional Kelly sizing
4) Almgren-Chriss style child-order slicing and shortfall tracking
5) Live broker integration hooks with paper-trading flag
6) Portfolio weight service (E2E Markowitz / HRP weights from Phase 3)
7) Real-time WebSocket market feed (Phase 4)
8) IDX ARB-avoidance execution override (Phase 4 IDX)
9) IDX broker API integration hooks (Phase 5)
"""

from __future__ import annotations

import math
import os
import time
import argparse
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import joblib
import pandas as pd
import backtrader as bt
import requests

from notifications import WebhookNotifier
from preflight_warmup import PreflightConfig, PreflightWarmup

try:
    from phase4_idx_arb_execution import (
        IDXAlmgrenChrissPlanner,
        ARBProbabilityMonitor,
        ARBMonitorConfig,
    )
except Exception:  # pragma: no cover - optional runtime dependency
    IDXAlmgrenChrissPlanner = None
    ARBProbabilityMonitor = None
    ARBMonitorConfig = None

try:
    from phase5_idx_broker_api import (
        IDXDataFeedWebSocket,
        IDXDataFeedConfig,
        IDXOrderBookLevel2,
    )
except Exception:  # pragma: no cover - optional runtime dependency
    IDXDataFeedWebSocket = None
    IDXDataFeedConfig = None
    IDXOrderBookLevel2 = None

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"
_DEFAULT_DATA_PATH = str(_SHARED / "data" / "phase7_trading_input_example.csv")
_DEFAULT_MASTER_DATA = str(_SHARED / "data" / "phase6_lpa_enriched.csv")
_DEFAULT_PROD_DIR = str(_SHARED / "models")
_DEFAULT_LOG = str(_SHARED / "logs" / "phase1_train_log.jsonl")
_DEFAULT_REPORT = str(_SHARED / "logs" / "PREFLIGHT_WARMUP_REPORT.md")


# ---------------------------
# Phase 3: Portfolio weight service
# ---------------------------
class PortfolioWeightService:
    """
    Loads E2E Markowitz or HRP portfolio weights from the shared model store.
    Used by the execution strategy to apply cross-asset optimal sizing.

    Priority resolution at runtime:
      1. E2E optimizer weights (e2e_portfolio_meta.pkl)
      2. HRP weights fallback (hrp_weights.pkl)
      3. Equal-weight distribution if no saved weights exist
    """

    def __init__(self, prod_dir: Optional[str] = None, ticker_suffix: Optional[str] = None):
        self.prod_dir = prod_dir or os.getenv("PROD_DIR", _DEFAULT_PROD_DIR)
        # Exchange-specific ticker suffix stripped when looking up weights.
        # Defaults to TICKER_SUFFIX env var, then ".JK" (IDX exchange).
        self.ticker_suffix = ticker_suffix or os.getenv("TICKER_SUFFIX", ".JK")
        self._weights: Optional[Dict[str, float]] = None
        self._asset_names: Optional[List[str]] = None
        self._source: str = "not_loaded"

    def load(self) -> bool:
        """Load weights. Returns True if successfully loaded from disk."""
        # Try E2E meta first (asset_names list)
        meta_path = os.path.join(self.prod_dir, "e2e_portfolio_meta.pkl")
        hrp_path = os.path.join(self.prod_dir, "hrp_weights.pkl")
        try:
            if os.path.exists(meta_path):
                meta = joblib.load(meta_path)
                self._asset_names = meta.get("asset_names", [])
                num_assets = len(self._asset_names)
                if num_assets > 0:
                    self._weights = {a: 1.0 / num_assets for a in self._asset_names}
                    self._source = "e2e_equal_fallback"
            if os.path.exists(hrp_path):
                saved = joblib.load(hrp_path)
                hrp_w = saved.get("weights")
                if hrp_w is not None and hasattr(hrp_w, "to_dict"):
                    self._weights = hrp_w.to_dict()
                    self._source = "hrp"
                    return True
        except Exception:
            pass
        return self._weights is not None

    def get_weight(self, symbol: str, default: float = 0.0) -> float:
        """Return portfolio weight for a symbol (e.g. 'BBRI' or 'BBRI.JK')."""
        if self._weights is None:
            self.load()
        if self._weights is None:
            return default
        company = symbol.replace(self.ticker_suffix, "")
        return float(self._weights.get(company, self._weights.get(symbol, default)))

    def get_max_position_fraction(self, symbol: str, max_cap: float = 0.25) -> float:
        """Convert portfolio weight to a max_pos_size cap for RiskManager."""
        w = self.get_weight(symbol)
        # Clamp to [0.01, max_cap] to avoid very small or very large positions
        return max(0.01, min(w if w > 0 else max_cap, max_cap))

    @property
    def source(self) -> str:
        return self._source


# ---------------------------
# Phase 4: Real-time WebSocket market feed (thin wrapper)
# ---------------------------
try:
    from phase4_distributed_pipeline import WebSocketMarketFeed, WebSocketConfig
except ImportError:  # pragma: no cover
    WebSocketMarketFeed = None  # type: ignore[assignment]
    WebSocketConfig = None  # type: ignore[assignment]


# ---------------------------
# Custom feed with extra lines
# ---------------------------
class CustomDataFeed(bt.feeds.PandasData):
    """
    Extend Backtrader PandasData so strategy can access:
    - finbert_score
    - pred_volatility
    - lpa_profile_id
    """

    lines = ("finbert_score", "pred_volatility", "lpa_profile_id", "is_pom_pom_regime")
    params = (
        ("finbert_score", -1),
        ("pred_volatility", -1),
        ("lpa_profile_id", -1),
        ("is_pom_pom_regime", -1),
    )


# ---------------------------
# Regime state machine
# ---------------------------
@dataclass
class RegimeConfig:
    action: str
    max_pos_size: float
    c_scale: float
    k_stop_mult: float


class RegimeStateMachine:
    """
    Maps LPA profile IDs to execution/risk states.
    """

    BREAKOUT_VOLATILE = {4, 6}
    TRENDING_NORMAL = {1, 3}
    MEAN_REVERSION = {7, 8}
    # "POM_POM" = pump-and-dump-like retail-hype regime on IDX.
    PUMP_DUMP_STOP_MULT = 0.6

    def resolve(self, profile_id: int, pom_pom_active: bool = False) -> RegimeConfig:
        if pom_pom_active:
            return RegimeConfig(
                action="PUMP_DUMP_RISK_OFF",
                max_pos_size=0.0,
                c_scale=0.0,
                k_stop_mult=self.PUMP_DUMP_STOP_MULT,
            )
        if profile_id in self.BREAKOUT_VOLATILE:
            return RegimeConfig(
                action="AGGRESSIVE_DIRECTIONAL",
                max_pos_size=0.05,
                c_scale=0.25,
                k_stop_mult=2.0,
            )
        if profile_id in self.TRENDING_NORMAL:
            return RegimeConfig(
                action="ALMGREN_CHRISS_EXECUTION",
                max_pos_size=0.15,
                c_scale=0.50,
                k_stop_mult=1.5,
            )
        if profile_id in self.MEAN_REVERSION:
            return RegimeConfig(
                action="PASSIVE_LIQUIDITY",
                max_pos_size=0.25,
                c_scale=1.00,
                k_stop_mult=1.0,
            )
        # fallback: conservative
        return RegimeConfig(
            action="PASSIVE_LIQUIDITY",
            max_pos_size=0.10,
            c_scale=0.30,
            k_stop_mult=1.2,
        )


# ---------------------------
# Risk manager
# ---------------------------
class RiskManager:
    """
    Implements:
    1) Vol-targeted stop distance:
       SL_dist = k * pred_volatility
    2) Fractional Kelly sizing:
       f* = (b p - q) / b, q = 1 - p
       Q_t = floor( PV_t / M * c f* * sigma_target / pred_volatility )
    """

    def __init__(self, sigma_target: float = 0.02, default_b: float = 1.2, default_p: float = 0.54):
        self.sigma_target = sigma_target
        self.default_b = default_b
        self.default_p = default_p

    @staticmethod
    def _safe(x: float, floor: float = 1e-6) -> float:
        if x is None or math.isnan(x) or x <= 0:
            return floor
        return x

    def stop_distance(self, k_mult: float, pred_volatility: float) -> float:
        pred_vol = self._safe(pred_volatility, floor=1e-4)
        return k_mult * pred_vol

    def kelly_fraction(self, p: float | None = None, b: float | None = None) -> float:
        p = self.default_p if p is None else p
        b = self.default_b if b is None else b
        q = 1 - p
        # f* = (b p - q) / b
        f_star = ((b * p) - q) / max(b, 1e-6)
        return max(0.0, min(f_star, 1.0))

    def target_size(
        self,
        portfolio_value: float,
        price: float,
        pred_volatility: float,
        c_scale: float,
        max_pos_size: float,
        kelly_p: float | None = None,
        kelly_b: float | None = None,
    ) -> int:
        if price <= 0:
            return 0

        f_star = self.kelly_fraction(p=kelly_p, b=kelly_b)
        pred_vol = self._safe(pred_volatility, floor=1e-4)
        M = price  # cash per unit

        # Q_t = floor( PV/M * c f* * sigma_target/pred_volatility )
        raw_q = (portfolio_value / M) * (c_scale * f_star) * (self.sigma_target / pred_vol)
        q = int(max(0, math.floor(raw_q)))

        max_q = int(math.floor((portfolio_value * max_pos_size) / M))
        return max(0, min(q, max_q))


# ---------------------------
# Almgren-Chriss planner
# ---------------------------
class AlmgrenChrissPlanner:
    """
    Generates child-order schedule for parent delta using a smooth
    hyperbolic-sine trajectory to reduce market impact.
    """

    def __init__(self, n_slices: int = 6, kappa: float = 1.0):
        self.n_slices = max(2, n_slices)
        self.kappa = max(0.1, kappa)

    def slice_delta(self, parent_delta: int) -> List[int]:
        if parent_delta == 0:
            return []
        sign = 1 if parent_delta > 0 else -1
        qty = abs(parent_delta)

        # Almgren-Chriss-like weight curve:
        # w_t ∝ sinh(kappa * (T - t))
        T = self.n_slices
        raw = [math.sinh(self.kappa * (T - t) / T) for t in range(1, T + 1)]
        s = sum(raw)
        weights = [x / s for x in raw]
        chunks = [int(math.floor(qty * w)) for w in weights]
        # distribute remainder
        rem = qty - sum(chunks)
        for i in range(rem):
            chunks[i % len(chunks)] += 1
        return [sign * c for c in chunks if c != 0]


@dataclass
class ParentOrderContext:
    parent_target: int
    decision_price: float
    stop_distance: float
    filled_size: int = 0
    fill_value: float = 0.0


class ExponentialBackoff:
    def __init__(self, base_seconds: float = 2.0, max_seconds: float = 60.0):
        self.base = base_seconds
        self.max = max_seconds
        self.attempt = 0

    def next_sleep(self) -> float:
        wait = min(self.max, self.base * (2 ** self.attempt))
        self.attempt += 1
        return wait

    def reset(self):
        self.attempt = 0


class LiveFeatureService:
    """
    Fetches latest FinBERT / pred_volatility / LPA profile for a live symbol.
    Priority:
    1) LIVE_FEATURES_API endpoint (if set)
    2) local features CSV path (LIVE_FEATURES_PATH)
    """

    def __init__(self, local_path: Optional[str] = None, api_url: Optional[str] = None):
        self.local_path = (local_path or os.getenv("LIVE_FEATURES_PATH", "")).strip()
        self.api_url = (api_url or os.getenv("LIVE_FEATURES_API", "")).strip()
        self._cache = {}

    def get_latest(self, symbol: str, dt: pd.Timestamp) -> Dict[str, float]:
        key = (symbol, pd.Timestamp(dt).floor("min"))
        if key in self._cache:
            return self._cache[key]

        out = {"finbert_score": 0.0, "pred_volatility": 0.02, "lpa_profile_id": 1, "is_pom_pom_regime": 0}
        try:
            if self.api_url:
                r = requests.get(
                    self.api_url,
                    params={"symbol": symbol, "ts": pd.Timestamp(dt).isoformat()},
                    timeout=5,
                )
                if r.status_code < 300:
                    payload = r.json()
                    out.update(
                        {
                            "finbert_score": float(payload.get("finbert_score", out["finbert_score"])),
                            "pred_volatility": float(payload.get("pred_volatility", out["pred_volatility"])),
                            "lpa_profile_id": int(payload.get("lpa_profile_id", out["lpa_profile_id"])),
                            "is_pom_pom_regime": int(payload.get("is_pom_pom_regime", out["is_pom_pom_regime"])),
                        }
                    )
                    self._cache[key] = out
                    return out
        except Exception:
            pass

        try:
            if self.local_path and os.path.exists(self.local_path):
                df = pd.read_csv(self.local_path)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                if "symbol" in df.columns:
                    sub = df[df["symbol"] == symbol].copy()
                elif "company" in df.columns:
                    sub = df[df["company"] == symbol.replace(".JK", "")].copy()
                else:
                    sub = df.copy()
                if not sub.empty and "date" in sub.columns:
                    sub = sub[sub["date"] <= pd.Timestamp(dt)].sort_values("date")
                    if not sub.empty:
                        row = sub.iloc[-1]
                        out.update(
                            {
                                "finbert_score": float(pd.to_numeric(row.get("finbert_score", out["finbert_score"]), errors="coerce") or 0.0),
                                "pred_volatility": float(pd.to_numeric(row.get("pred_volatility", row.get("volatility_20d", out["pred_volatility"])), errors="coerce") or 0.02),
                                "lpa_profile_id": int(pd.to_numeric(row.get("lpa_profile_id", out["lpa_profile_id"]), errors="coerce") or 1),
                                "is_pom_pom_regime": int(pd.to_numeric(row.get("is_pom_pom_regime", out["is_pom_pom_regime"]), errors="coerce") or 0),
                            }
                        )
        except Exception:
            pass

        self._cache[key] = out
        return out


# ---------------------------
# Main strategy
# ---------------------------
class RegimeExecutionStrategy(bt.Strategy):
    params = dict(
        sigma_target=0.02,
        kelly_p=0.54,
        kelly_b=1.2,
        ac_slices=6,
        ac_kappa=1.0,
        min_rebalance_delta=5,
        notifier=None,
        feature_service=None,
        live_symbol="",
        portfolio_weight_service=None,
        idx_data_feed=None,
        arb_monitor_cfg=None,
    )

    def __init__(self):
        self.state_machine = RegimeStateMachine()
        self.risk = RiskManager(
            sigma_target=self.p.sigma_target,
            default_b=self.p.kelly_b,
            default_p=self.p.kelly_p,
        )
        # Phase 4 IDX: use ARB-aware planner when available, fall back to standard
        if IDXAlmgrenChrissPlanner is not None:
            mon_cfg = self.p.arb_monitor_cfg or ARBMonitorConfig()
            self.ac = IDXAlmgrenChrissPlanner(
                n_slices=self.p.ac_slices,
                kappa=self.p.ac_kappa,
                monitor_cfg=mon_cfg,
            )
        else:
            self.ac = AlmgrenChrissPlanner(n_slices=self.p.ac_slices, kappa=self.p.ac_kappa)

        # Phase 5 IDX: LOB data feed
        self.idx_feed: Optional[object] = self.p.idx_data_feed

        self.pending_child_orders = deque()
        self.parent_ctx: Optional[ParentOrderContext] = None
        self.active_stop_order = None
        self.last_parent_ref = None

        # Metrics
        self.shortfall_records: List[Tuple[pd.Timestamp, float]] = []
        self.notifier: Optional[WebhookNotifier] = self.p.notifier
        self.feature_service: Optional[LiveFeatureService] = self.p.feature_service
        self.portfolio_weight_svc: Optional[PortfolioWeightService] = self.p.portfolio_weight_service
        self.live_symbol = self.p.live_symbol
        self.last_profile_id: Optional[int] = None
        self.last_pom_pom_active: bool = False

    def _send_alert(self, event: str, message: str, payload: Optional[dict] = None):
        if self.notifier:
            self.notifier.send(event=event, message=message, payload=payload)

    def _signal_direction(self, finbert: float) -> int:
        """
        Simple directional proxy for execution decisions:
        +1 for bullish sentiment, -1 for bearish sentiment, 0 neutral.
        """
        if finbert > 0.10:
            return 1
        if finbert < -0.10:
            return -1
        return 0

    def _cancel_stop(self):
        if self.active_stop_order is not None:
            self.cancel(self.active_stop_order)
            self.active_stop_order = None

    def _place_dynamic_stop(self, stop_dist: float):
        """
        Bind stop-loss to active position.
        stop_dist is volatility-scaled fraction (k * pred_volatility).
        """
        pos = self.getposition(self.data).size
        if pos == 0:
            self._cancel_stop()
            return
        self._cancel_stop()
        px = float(self.data.close[0])
        if pos > 0:
            stop_px = px * (1.0 - stop_dist)
            self.active_stop_order = self.sell(exectype=bt.Order.Stop, price=stop_px, size=abs(pos))
        else:
            stop_px = px * (1.0 + stop_dist)
            self.active_stop_order = self.buy(exectype=bt.Order.Stop, price=stop_px, size=abs(pos))

    def _execute_parent_delta(
        self,
        parent_delta: int,
        decision_price: float,
        stop_dist: float,
        use_slicing: bool,
        kelly_fraction: float,
        target_abs_size: int,
        regime_action: str,
        pred_volatility: float = 0.02,
        bars_elapsed: int = 0,
    ):
        if parent_delta == 0:
            return
        self.parent_ctx = ParentOrderContext(
            parent_target=parent_delta,
            decision_price=decision_price,
            stop_distance=stop_dist,
        )
        if use_slicing:
            # Phase 4 IDX: use ARB-aware slice when the planner supports it
            if IDXAlmgrenChrissPlanner is not None and isinstance(self.ac, IDXAlmgrenChrissPlanner):
                chunks = self.ac.slice_delta_arb_aware(
                    parent_delta,
                    current_price=decision_price,
                    pred_daily_volatility=pred_volatility,
                    bars_elapsed=bars_elapsed,
                )
            else:
                chunks = self.ac.slice_delta(parent_delta)
            for q in chunks:
                self.pending_child_orders.append(q)
        else:
            self.pending_child_orders.append(parent_delta)
        self._send_alert(
            "parent_order_initiated",
            f"Parent delta={parent_delta}, kelly={kelly_fraction:.4f}, stop_dist={stop_dist:.5f}",
            {
                "parent_delta": parent_delta,
                "regime_action": regime_action,
                "kelly_fraction": kelly_fraction,
                "target_abs_size": target_abs_size,
                "decision_price": decision_price,
                "stop_distance": stop_dist,
                "pending_children": len(self.pending_child_orders),
            },
        )

    def _dispatch_next_child(self):
        if not self.pending_child_orders:
            return
        q = self.pending_child_orders.popleft()
        if q > 0:
            self.buy(size=abs(q))
        else:
            self.sell(size=abs(q))

    def next(self):
        # Never use future values: all features are read at index [0] only (current bar).
        dt = self.data.datetime.datetime(0)

        def read_feed(name: str, default: float):
            try:
                v = float(getattr(self.data, name)[0])
                if math.isnan(v):
                    return default
                return v
            except Exception:
                return default

        profile_id = int(read_feed("lpa_profile_id", 1))
        pred_vol = read_feed("pred_volatility", 0.02)
        finbert = read_feed("finbert_score", 0.0)
        pom_pom_active = int(read_feed("is_pom_pom_regime", 0)) == 1

        if self.feature_service is not None:
            live_sym = self.live_symbol or ""
            snap = self.feature_service.get_latest(symbol=live_sym, dt=pd.Timestamp(dt))
            # In live mode, feature service values override feed extras.
            finbert = float(snap.get("finbert_score", finbert))
            pred_vol = float(snap.get("pred_volatility", pred_vol))
            profile_id = int(snap.get("lpa_profile_id", profile_id))
            pom_pom_active = int(snap.get("is_pom_pom_regime", int(pom_pom_active))) == 1

        close_px = float(self.data.close[0])
        pv = float(self.broker.getvalue())

        # Phase 5 IDX: pull LOB imbalance from the live IDX data feed when available
        lob_imbalance = 0.0
        if self.idx_feed is not None and IDXDataFeedWebSocket is not None:
            live_sym = self.live_symbol or ""
            if live_sym:
                lob_imbalance = float(self.idx_feed.get_lob_imbalance(live_sym))

        regime = self.state_machine.resolve(profile_id, pom_pom_active=pom_pom_active)
        stop_dist = self.risk.stop_distance(regime.k_stop_mult, pred_vol)

        if self.last_profile_id != profile_id:
            self._send_alert(
                "regime_change",
                f"LPA regime changed to {profile_id} ({regime.action})",
                {"profile_id": profile_id, "action": regime.action, "timestamp": pd.Timestamp(dt).isoformat()},
            )
            self.last_profile_id = profile_id
        if pom_pom_active and not self.last_pom_pom_active:
            self._send_alert(
                "pom_pom_regime_active",
                "POM_POM regime active: forcing 0% allocation and tighter stops.",
                {"profile_id": profile_id, "timestamp": pd.Timestamp(dt).isoformat()},
            )
        self.last_pom_pom_active = bool(pom_pom_active)

        direction = self._signal_direction(finbert)
        f_star = self.risk.kelly_fraction(p=self.p.kelly_p, b=self.p.kelly_b)

        # Phase 3: use portfolio weight service to constrain max position size
        # when E2E Markowitz or HRP weights are available.
        effective_max_pos = regime.max_pos_size
        if self.portfolio_weight_svc is not None:
            sym = self.live_symbol or ""
            effective_max_pos = self.portfolio_weight_svc.get_max_position_fraction(
                sym, max_cap=regime.max_pos_size
            )

        target_abs = self.risk.target_size(
            portfolio_value=pv,
            price=close_px,
            pred_volatility=pred_vol,
            c_scale=regime.c_scale,
            max_pos_size=effective_max_pos,
            kelly_p=self.p.kelly_p,
            kelly_b=self.p.kelly_b,
        )
        target_pos = direction * target_abs

        current_pos = int(self.getposition(self.data).size)
        delta = target_pos - current_pos

        # Dispatch existing child order queue first
        if self.pending_child_orders:
            self._dispatch_next_child()
            return

        if abs(delta) < self.p.min_rebalance_delta:
            self._place_dynamic_stop(stop_dist)
            return

        if regime.action == "ALMGREN_CHRISS_EXECUTION":
            self._execute_parent_delta(
                delta,
                decision_price=close_px,
                stop_dist=stop_dist,
                use_slicing=True,
                kelly_fraction=f_star,
                target_abs_size=target_abs,
                regime_action=regime.action,
                pred_volatility=pred_vol,
            )
        elif regime.action in {"AGGRESSIVE_DIRECTIONAL", "PUMP_DUMP_RISK_OFF"}:
            self._execute_parent_delta(
                delta,
                decision_price=close_px,
                stop_dist=stop_dist,
                use_slicing=False,
                kelly_fraction=f_star,
                target_abs_size=target_abs,
                regime_action=regime.action,
                pred_volatility=pred_vol,
            )
        else:  # PASSIVE_LIQUIDITY
            # Passive mode: only move part of the required delta each step
            passive_delta = int(delta * 0.5)
            if passive_delta == 0:
                passive_delta = 1 if delta > 0 else -1
            self._execute_parent_delta(
                passive_delta,
                decision_price=close_px,
                stop_dist=stop_dist,
                use_slicing=True,
                kelly_fraction=f_star,
                target_abs_size=target_abs,
                regime_action=regime.action,
                pred_volatility=pred_vol,
            )

        self._send_alert(
            "execution_decision",
            "Execution decision computed",
            {
                "profile_id": profile_id,
                "action": regime.action,
                "kelly_fraction": f_star,
                "target_abs_size": target_abs,
                "delta": delta,
                "pred_volatility": pred_vol,
                "stop_distance": stop_dist,
                "is_pom_pom_regime": int(pom_pom_active),
                "lob_imbalance": float(lob_imbalance),
            },
        )
        self._dispatch_next_child()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            return
        if order.status == order.Completed and order.exectype == bt.Order.Stop:
            self._send_alert(
                "stop_triggered",
                "Stop-loss order executed",
                {
                    "fill_price": float(order.executed.price),
                    "fill_size": int(order.executed.size),
                },
            )
        if order.status == order.Completed and self.parent_ctx is not None:
            size = int(order.executed.size)
            px = float(order.executed.price)
            self.parent_ctx.filled_size += size
            self.parent_ctx.fill_value += px * size

            # Implementation shortfall against decision price
            # Positive value means worse execution (cost) for buy; mirrored for sells by sign(size).
            shortfall = (px - self.parent_ctx.decision_price) * size
            dt = self.data.datetime.datetime(0)
            self.shortfall_records.append((dt, shortfall))

            # If no more child orders pending, treat parent complete and bind stop.
            if not self.pending_child_orders:
                self._place_dynamic_stop(self.parent_ctx.stop_distance)
                self._send_alert(
                    "order_filled",
                    f"Parent completed at avg fill component {px:.4f}",
                    {
                        "filled_size": self.parent_ctx.filled_size,
                        "last_fill_price": px,
                        "stop_distance": self.parent_ctx.stop_distance,
                    },
                )
                self.parent_ctx = None

    def stop(self):
        if self.shortfall_records:
            total_shortfall = sum(v for _, v in self.shortfall_records)
            self.log(f"Total implementation shortfall: {total_shortfall:.4f}")

    def log(self, txt):
        dt = self.data.datetime.datetime(0)
        print(f"{dt.isoformat()} | {txt}")


# ---------------------------
# Engine runner
# ---------------------------
def run_backtest(
    data_path: str,
    start_cash: float = 1_000_000.0,
    commission: float = 0.001,
) -> Dict[str, float]:
    """
    Backtest runner for prepared dataset containing custom columns.
    Required columns:
      datetime index or date column + OHLCV + finbert_score + pred_volatility + lpa_profile_id
    """
    df = pd.read_csv(data_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date").set_index("date")
    required = ["open", "high", "low", "close", "volume", "finbert_score", "pred_volatility", "lpa_profile_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cerebro = bt.Cerebro(stdstats=False)
    data = CustomDataFeed(dataname=df)
    cerebro.adddata(data)
    cerebro.addstrategy(RegimeExecutionStrategy)
    cerebro.broker.setcash(start_cash)
    cerebro.broker.setcommission(commission=commission)

    start_value = cerebro.broker.getvalue()
    results = cerebro.run()
    end_value = cerebro.broker.getvalue()
    pnl = end_value - start_value
    return {
        "start_value": float(start_value),
        "end_value": float(end_value),
        "pnl": float(pnl),
        "return_pct": float((end_value / start_value - 1.0) * 100.0),
        "strategy_instances": len(results),
    }


def run_live_ib(
    symbol: str,
    timeframe: int = bt.TimeFrame.Minutes,
    compression: int = 1,
    paper: bool = True,
    notifier: Optional[WebhookNotifier] = None,
    feature_service: Optional[LiveFeatureService] = None,
) -> None:
    """
    Live execution using Backtrader IBStore.
    Paper mode uses paper port by default (IB_PAPER_PORT, fallback 7497).
    Live mode uses IB_LIVE_PORT (fallback 7496).
    """
    try:
        IBStore = bt.stores.IBStore
    except Exception as e:
        raise RuntimeError("IBStore is unavailable in this Backtrader installation.") from e

    host = os.getenv("IB_HOST", "127.0.0.1")
    port = int(os.getenv("IB_PAPER_PORT", "7497") if paper else os.getenv("IB_LIVE_PORT", "7496"))
    client_id = int(os.getenv("IB_CLIENT_ID", "101"))

    store = IBStore(host=host, port=port, clientId=client_id, reconnect=True, timeout=3.0)

    cerebro = bt.Cerebro(stdstats=False, quicknotify=True)
    data = store.getdata(
        dataname=symbol,
        timeframe=timeframe,
        compression=compression,
        historical=False,
        backfill_start=True,
        backfill=True,
        rtbar=True,
        qcheck=1.0,
    )
    broker = store.getbroker()
    cerebro.setbroker(broker)
    cerebro.adddata(data)
    cerebro.addstrategy(
        RegimeExecutionStrategy,
        notifier=notifier,
        feature_service=feature_service,
        live_symbol=symbol,
    )
    if notifier:
        notifier.send(
            "live_start",
            f"Starting live {'paper' if paper else 'real'} trading for {symbol}",
            {"symbol": symbol, "paper": paper, "host": host, "port": port},
        )
    cerebro.run()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trading engine (backtest/live)")
    p.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    p.add_argument("--broker", choices=["ib"], default="ib", help="Live broker store backend")
    p.add_argument("--paper", action="store_true", help="Use paper trading environment")
    p.add_argument("--data-path", default=_DEFAULT_DATA_PATH)
    p.add_argument("--symbol", default=os.getenv("LIVE_SYMBOL", "BBRI-STK-SMART-USD"))
    p.add_argument("--run-preflight", action="store_true", help="Run synchronous pre-flight warm-up before live connect")
    p.add_argument("--preflight-master-data", default=os.getenv("PREFLIGHT_MASTER_DATA", _DEFAULT_MASTER_DATA))
    p.add_argument("--preflight-prod-dir", default=os.getenv("PREFLIGHT_PROD_DIR", _DEFAULT_PROD_DIR))
    p.add_argument("--preflight-log-path", default=os.getenv("PREFLIGHT_LOG_PATH", _DEFAULT_LOG))
    p.add_argument("--preflight-report", default=os.getenv("PREFLIGHT_REPORT_PATH", _DEFAULT_REPORT))
    p.add_argument("--preflight-lookback-days", type=int, default=int(os.getenv("PREFLIGHT_LOOKBACK_DAYS", "90")))
    p.add_argument("--preflight-warmup-epochs", type=int, default=int(os.getenv("PREFLIGHT_WARMUP_EPOCHS", "6")))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    notifier = WebhookNotifier()
    feature_service = LiveFeatureService()

    if args.mode == "backtest":
        if not pd.io.common.file_exists(args.data_path):
            print(
                "Provide an execution dataset with columns: "
                "date, open, high, low, close, volume, finbert_score, pred_volatility, lpa_profile_id."
            )
        else:
            stats = run_backtest(args.data_path)
            print(stats)
    else:
        if args.broker != "ib":
            raise ValueError("Current live implementation supports broker='ib' for Backtrader store integration.")

        if args.run_preflight:
            cfg = PreflightConfig(
                master_data=args.preflight_master_data,
                prod_dir=args.preflight_prod_dir,
                log_path=args.preflight_log_path,
                report_path=args.preflight_report,
                lookback_days=args.preflight_lookback_days,
                warmup_epochs=args.preflight_warmup_epochs,
            )
            preflight = PreflightWarmup(cfg)
            if notifier:
                notifier.send("preflight_start", "Starting synchronous pre-flight warm-up", {"master_data": args.preflight_master_data})
            preflight_result = preflight.run()
            if notifier:
                notifier.send("preflight_ok", "Pre-flight completed; proceeding to live connection", preflight_result)

        backoff = ExponentialBackoff(base_seconds=2.0, max_seconds=120.0)
        while True:
            try:
                run_live_ib(
                    symbol=args.symbol,
                    paper=args.paper,
                    notifier=notifier,
                    feature_service=feature_service,
                )
                backoff.reset()
            except KeyboardInterrupt:
                print("Live engine stopped by user.")
                break
            except Exception as exc:
                wait = backoff.next_sleep()
                print(f"[LIVE ERROR] {exc}. Reconnecting in {wait:.1f}s")
                if notifier:
                    notifier.send("live_reconnect", "Broker disconnected; reconnecting", {"error": str(exc), "wait_seconds": wait})
                time.sleep(wait)
