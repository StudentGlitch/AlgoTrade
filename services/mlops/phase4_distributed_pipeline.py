"""
Phase 4: Distributed Training & Real-Time Execution
-----------------------------------------------------
RayDistributedPipeline:
  - Wraps the train_pipeline using Ray for local distributed computing.
  - Distributes per-company data preprocessing across CPU workers.
  - Provides a hook for hyperparameter search via ray.tune.
  - Requires ray[default]>=2.52.0 (security fix for DNS rebinding attack).
  - NOTE: Run only on trusted local/private networks. Do not expose the Ray
    dashboard to the public internet; the dashboard endpoint has known
    unauthenticated access vulnerabilities (CVE details: ray advisory DB).

WebSocketMarketFeed:
  - Real-time tick data ingestor using a WebSocket connection
    (e.g. Polygon.io, dxFeed, or any generic OHLCV WebSocket source).
  - Thread-safe: tick queue consumed by Backtrader or other strategies.
  - Sub-second state update latency for 800+ assets.
  - Falls back gracefully when no API key is configured.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import ray
except ImportError:  # pragma: no cover
    ray = None

try:
    import websocket  # websocket-client package
except ImportError:  # pragma: no cover
    websocket = None

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"

POLYGON_WS_URL = "wss://socket.polygon.io/stocks"

# ---------------------------------------------------------------------------
# Ray remote preprocessing task
# ---------------------------------------------------------------------------

def _preprocess_one_company(  # Ray remote worker (defined as plain function)
    company: str,
    records: list,
    features: list[str],
    lookback: int,
) -> dict:
    """
    Preprocess a single company's data slice.
    Returns serialisable dict suitable for aggregation.
    """
    df = pd.DataFrame(records)
    if df.empty or "date" not in df.columns:
        return {"company": company, "n_sequences": 0, "rows": []}

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    for c in features + ["abs_return"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    avail_features = [c for c in features if c in df.columns]
    df[avail_features] = df[avail_features].ffill().bfill()
    df = df.dropna(subset=avail_features + ["abs_return"])

    sequences = []
    vals = df[avail_features].values.astype(np.float32)
    tgt = df["abs_return"].values.astype(np.float32)
    dates = df["date"].astype(str).values

    for i in range(lookback, len(df)):
        sequences.append({
            "company": company,
            "date": str(dates[i]),
            "seq": vals[i - lookback : i].tolist(),
            "target": float(tgt[i]),
        })

    return {"company": company, "n_sequences": len(sequences), "rows": sequences}


@dataclass
class RayConfig:
    num_cpus: Optional[int] = None
    lookback: int = 10
    batch_companies: int = 50
    features: list = None


class RayDistributedPipeline:
    """
    Distributes data preprocessing across Ray workers for sub-linear
    wall-clock time when processing 800+ assets.

    Usage:
        pipeline = RayDistributedPipeline(cfg)
        seqs = pipeline.distributed_preprocess(df, features)
    """

    FEATURES_DEFAULT = [
        "abs_return", "volatility_5d", "volatility_20d", "volume_z20",
        "range_pct", "finbert_score", "vix_close_ret", "usd_idr_close_ret",
    ]

    def __init__(self, cfg: Optional[RayConfig] = None):
        self.cfg = cfg or RayConfig()
        if self.cfg.features is None:
            self.cfg.features = self.FEATURES_DEFAULT
        self._ray_initialized = False

    def _ensure_ray(self) -> bool:
        if ray is None:
            return False
        if not self._ray_initialized:
            if not ray.is_initialized():
                init_kwargs = {"ignore_reinit_error": True, "include_dashboard": False}
                if self.cfg.num_cpus is not None:
                    init_kwargs["num_cpus"] = self.cfg.num_cpus
                ray.init(**init_kwargs)
            self._ray_initialized = True
        return True

    def distributed_preprocess(
        self, df: pd.DataFrame, features: Optional[list[str]] = None
    ) -> list[dict]:
        """
        Preprocess all companies in parallel using Ray remote tasks.
        Falls back to sequential processing when Ray is unavailable.

        Returns a flat list of sequence dicts with keys:
          company, date, seq (lookback × n_features), target.
        """
        features = features or self.cfg.features
        companies = sorted(df["company"].dropna().unique().tolist())
        results = []

        if self._ensure_ray():
            remote_fn = ray.remote(_preprocess_one_company)
            futures = []
            for company in companies:
                records = (
                    df[df["company"] == company]
                    .to_dict(orient="records")
                )
                futures.append(
                    remote_fn.remote(company, records, features, self.cfg.lookback)
                )
            for batch_start in range(0, len(futures), self.cfg.batch_companies):
                batch = futures[batch_start : batch_start + self.cfg.batch_companies]
                batch_results = ray.get(batch)
                for res in batch_results:
                    results.extend(res.get("rows", []))
        else:
            # Sequential fallback (single process)
            for company in companies:
                records = (
                    df[df["company"] == company]
                    .to_dict(orient="records")
                )
                res = _preprocess_one_company(company, records, features, self.cfg.lookback)
                results.extend(res.get("rows", []))

        return results

    def tune_hyperparameters(
        self,
        df: pd.DataFrame,
        train_cutoff: pd.Timestamp,
        param_space: Optional[dict] = None,
        num_samples: int = 20,
    ) -> dict:
        """
        Hyperparameter search for GlobalPanelLSTM using ray.tune.
        Falls back to returning the default config when ray.tune is absent.
        """
        try:
            from ray import tune as raytune
        except ImportError:
            return {
                "best_config": {
                    "lookback": 10,
                    "lstm_units_1": 64,
                    "lstm_units_2": 32,
                    "learning_rate": 1e-3,
                },
                "ray_tune_available": False,
            }

        from phase2_global_panel_lstm import GlobalPanelLSTMRefitter, PanelLSTMConfig

        if param_space is None:
            param_space = {
                "lookback": raytune.choice([5, 10, 15]),
                "lstm_units_1": raytune.choice([32, 64, 128]),
                "lstm_units_2": raytune.choice([16, 32, 64]),
                "learning_rate": raytune.loguniform(1e-4, 1e-2),
                "embedding_dim": raytune.choice([8, 16, 32]),
            }

        def _train_trial(config):
            cfg = PanelLSTMConfig(
                lookback=config["lookback"],
                lstm_units_1=config["lstm_units_1"],
                lstm_units_2=config["lstm_units_2"],
                learning_rate=config["learning_rate"],
                embedding_dim=config.get("embedding_dim", 16),
                epochs=10,
            )
            refitter = GlobalPanelLSTMRefitter(cfg)
            try:
                stats = refitter.fit(df.copy(), train_cutoff)
                raytune.report({"val_loss": stats.get("final_val_loss", 9999.0)})
            except Exception as exc:
                raytune.report({"val_loss": 9999.0, "error": str(exc)})

        self._ensure_ray()
        analysis = raytune.run(
            _train_trial,
            config=param_space,
            num_samples=num_samples,
            metric="val_loss",
            mode="min",
            verbose=0,
        )
        best = analysis.get_best_config(metric="val_loss", mode="min")
        return {"best_config": best, "ray_tune_available": True}

    def shutdown(self) -> None:
        if ray is not None and ray.is_initialized():
            ray.shutdown()
        self._ray_initialized = False


# ---------------------------------------------------------------------------
# WebSocket real-time market feed
# ---------------------------------------------------------------------------

@dataclass
class WebSocketConfig:
    api_key: str = os.getenv("POLYGON_API_KEY", "")
    url: str = POLYGON_WS_URL
    subscriptions: list = None
    reconnect_delay: float = 5.0
    max_queue_size: int = 100_000
    heartbeat_interval: float = 30.0


@dataclass
class MarketTick:
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0


class WebSocketMarketFeed:
    """
    Real-time market data feed via WebSocket (Polygon.io or compatible API).

    Connects, authenticates, and subscribes to configured symbols.
    Ticks are stored in a thread-safe queue; consumers call get_latest_tick()
    or drain_ticks() for sub-second market state updates.

    Falls back gracefully when POLYGON_API_KEY is absent (returns empty feed).
    """

    def __init__(self, cfg: Optional[WebSocketConfig] = None):
        self.cfg = cfg or WebSocketConfig()
        if self.cfg.subscriptions is None:
            self.cfg.subscriptions = []
        self._tick_queue: queue.Queue = queue.Queue(maxsize=self.cfg.max_queue_size)
        self._latest: dict[str, MarketTick] = {}
        self._lock = threading.Lock()
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._authenticated = False
        self._last_heartbeat = time.time()

    def enabled(self) -> bool:
        return bool(self.cfg.api_key and websocket is not None)

    def start(self) -> None:
        if not self.enabled():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass

    def _run_loop(self) -> None:
        while self._running:
            try:
                self._ws = websocket.WebSocketApp(
                    self.cfg.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._ws.run_forever(ping_interval=int(self.cfg.heartbeat_interval))
            except Exception:
                pass
            if self._running:
                time.sleep(self.cfg.reconnect_delay)

    def _on_open(self, ws) -> None:
        self._authenticated = False
        ws.send(json.dumps({"action": "auth", "params": self.cfg.api_key}))

    def _on_message(self, ws, raw_msg: str) -> None:
        try:
            messages = json.loads(raw_msg)
            if not isinstance(messages, list):
                messages = [messages]
            for msg in messages:
                ev = msg.get("ev", "")
                if ev == "status" and msg.get("status") == "auth_success":
                    self._authenticated = True
                    if self.cfg.subscriptions:
                        subs = ",".join(self.cfg.subscriptions)
                        ws.send(json.dumps({"action": "subscribe", "params": subs}))
                elif ev in ("T", "Q", "AM") and self._authenticated:
                    self._handle_tick(msg)
        except Exception:
            pass

    def _handle_tick(self, msg: dict) -> None:
        symbol = msg.get("sym", "") or msg.get("s", "")
        if not symbol:
            return
        price = float(msg.get("p", 0) or msg.get("vw", 0) or 0)
        volume = float(msg.get("s", 0) if "s" in msg and isinstance(msg["s"], (int, float)) else msg.get("av", 0) or 0)
        ts_ms = msg.get("t", msg.get("e", 0)) or 0
        ts = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc) if ts_ms else datetime.now(tz=timezone.utc)
        tick = MarketTick(
            symbol=symbol,
            timestamp=ts,
            price=price,
            volume=volume,
            bid=float(msg.get("bx", 0) or msg.get("bp", 0) or 0),
            ask=float(msg.get("ax", 0) or msg.get("ap", 0) or 0),
        )
        with self._lock:
            self._latest[symbol] = tick
        if not self._tick_queue.full():
            self._tick_queue.put_nowait(tick)

    def _on_error(self, ws, error) -> None:
        pass

    def _on_close(self, ws, close_status_code, close_msg) -> None:
        self._authenticated = False

    def get_latest_tick(self, symbol: str) -> Optional[MarketTick]:
        with self._lock:
            return self._latest.get(symbol)

    def drain_ticks(self, max_ticks: int = 1000) -> list[MarketTick]:
        ticks = []
        for _ in range(max_ticks):
            try:
                ticks.append(self._tick_queue.get_nowait())
            except queue.Empty:
                break
        return ticks

    def get_latest_prices(self) -> dict[str, float]:
        with self._lock:
            return {sym: t.price for sym, t in self._latest.items()}

    def subscribe(self, symbols: list[str]) -> None:
        """Dynamically subscribe to additional symbols (e.g. new IPOs)."""
        self.cfg.subscriptions.extend(symbols)
        if self._ws is not None and self._authenticated:
            params = ",".join(f"T.{s}" for s in symbols)
            try:
                self._ws.send(json.dumps({"action": "subscribe", "params": params}))
            except Exception:
                pass


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Phase 4: distributed pipeline + WebSocket feed test")
    p.add_argument("--data", default=str(_SHARED / "data" / "phase6_lpa_enriched.csv"))
    p.add_argument("--num-cpus", type=int, default=None)
    p.add_argument("--ws-test", action="store_true", help="Test WebSocket feed (requires POLYGON_API_KEY)")
    args = p.parse_args()

    if os.path.exists(args.data):
        df = pd.read_csv(args.data)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "company"]).sort_values(["company", "date"])

        cfg = RayConfig(num_cpus=args.num_cpus)
        pipeline = RayDistributedPipeline(cfg)
        seqs = pipeline.distributed_preprocess(df)
        print(f"PREPROCESSED={len(seqs)} sequences across {df['company'].nunique()} assets")
        pipeline.shutdown()

    if args.ws_test:
        feed_cfg = WebSocketConfig(subscriptions=["T.*"])
        feed = WebSocketMarketFeed(feed_cfg)
        if feed.enabled():
            print("WebSocket feed enabled. Starting for 5 seconds...")
            feed.start()
            time.sleep(5)
            ticks = feed.drain_ticks()
            print(f"TICKS_RECEIVED={len(ticks)}")
            feed.stop()
        else:
            print("WebSocket feed disabled (no POLYGON_API_KEY or websocket-client missing).")


if __name__ == "__main__":
    main()
