"""
Phase 4: IDX ARB-Avoidance Almgren-Chriss Execution Override
------------------------------------------------------------
Standard Almgren-Chriss assumes continuous two-sided liquidity.  On the IDX,
when a stock hits Auto Rejection Bawah (ARB) the bid side of the LOB empties
and the algorithm is trapped with an unsellable position.

This module adds:
  ARBProbabilityMonitor
    Monitors the distance between the current intraday price and the ARB
    floor.  Computes prob_hit_arb using the predicted volatility and remaining
    distance.  Uses a Gaussian CDF approximation:

       prob_hit_arb = Φ( (arb_floor - price) / (sigma * price * sqrt(T)) )

    where sigma is the predicted daily volatility and T is the remaining
    fraction of the trading day expressed in daily units.

  IDXAlmgrenChrissPlanner
    Subclass-compatible drop-in for AlmgrenChrissPlanner in trading_engine.py.
    When prob_hit_arb >= threshold the standard hyperbolic-sine trajectory is
    overridden with a *front-loaded* schedule that concentrates execution in
    the first slice(s) to guarantee liquidation before ARB locks the LOB.

    Front-loaded schedule  w_t ∝ exp(-lambda * t / T):
      high lambda → nearly all volume in first slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

try:
    from scipy.stats import norm as _scipy_norm
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# IDX tick sizes (approximate; first board is used here)
# ---------------------------------------------------------------------------

IDX_ARB_LIMITS_BY_PRICE = (
    # (max_price, arb_fraction)  — lower price bands have wider ARB limits
    (200,      -0.35),
    (5_000,    -0.25),
    (5_000_0,  -0.20),
    (float("inf"), -0.10),
)


def idx_arb_limit_for_price(price: float) -> float:
    """
    Return the ARB limit fraction for a given price level.

    IDX applies wider limits for cheaper stocks:
      price ≤ 200      → -35 %
      200 < price ≤ 5000   → -25 %
      5000 < price ≤ 50000 → -20 %
      price > 50000    → -10 %
    """
    for max_px, limit in IDX_ARB_LIMITS_BY_PRICE:
        if price <= max_px:
            return limit
    return -0.10


# ---------------------------------------------------------------------------
# ARB probability monitor
# ---------------------------------------------------------------------------

@dataclass
class ARBMonitorConfig:
    """Configuration for ARBProbabilityMonitor."""

    front_load_threshold: float = 0.20
    """prob_hit_arb at or above this value triggers front-loaded execution."""

    front_load_lambda: float = 4.0
    """Exponential decay rate for the front-loaded weight curve."""

    daily_sessions: int = 480
    """Approximate number of 1-minute bars per IDX trading day."""

    arb_limit_override: Optional[float] = None
    """If set, use this constant ARB limit fraction instead of price-based lookup."""


class ARBProbabilityMonitor:
    """
    Estimates the probability that a position will be trapped by an ARB halt.

    Parameters
    ----------
    cfg : ARBMonitorConfig

    Usage
    -----
    monitor = ARBProbabilityMonitor()
    prob = monitor.compute_prob_hit_arb(
        current_price=1500.0,
        pred_daily_volatility=0.04,
        bars_elapsed=120,          # minutes since market open
    )
    """

    def __init__(self, cfg: Optional[ARBMonitorConfig] = None):
        self.cfg = cfg or ARBMonitorConfig()

    def _arb_floor(self, price: float) -> float:
        if self.cfg.arb_limit_override is not None:
            return price * (1.0 + self.cfg.arb_limit_override)
        return price * (1.0 + idx_arb_limit_for_price(price))

    def compute_prob_hit_arb(
        self,
        current_price: float,
        pred_daily_volatility: float,
        bars_elapsed: int = 0,
    ) -> float:
        """
        Gaussian approximation of the probability that price will reach the
        ARB floor before the end of the trading day.

        Args:
            current_price: Most recent trade price (IDX: Rupiah).
            pred_daily_volatility: LSTM-predicted daily volatility (fractional,
                e.g. 0.03 = 3 %).
            bars_elapsed: Number of 1-minute bars elapsed since market open.
                Used to scale the remaining time horizon T.

        Returns:
            Probability in [0, 1].
        """
        if current_price <= 0 or pred_daily_volatility <= 0:
            return 0.0

        arb_floor = self._arb_floor(current_price)
        distance = (arb_floor - current_price) / current_price  # negative

        remaining = max(
            1, self.cfg.daily_sessions - max(0, bars_elapsed)
        )
        t_frac = remaining / float(self.cfg.daily_sessions)
        sigma_horizon = pred_daily_volatility * math.sqrt(t_frac)

        if sigma_horizon <= 0:
            return 0.0

        z = distance / sigma_horizon  # distance is negative → z < 0

        if _HAS_SCIPY:
            return float(_scipy_norm.cdf(z))

        # Fallback: rational approximation to Φ(z) for z < 0
        # Abramowitz & Stegun 26.2.17
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        phi_z = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
        if z < 0:
            return phi_z * poly
        return 1.0 - phi_z * poly

    def is_arb_risk_elevated(
        self,
        current_price: float,
        pred_daily_volatility: float,
        bars_elapsed: int = 0,
    ) -> bool:
        """Return True when ARB probability exceeds the configured threshold."""
        prob = self.compute_prob_hit_arb(current_price, pred_daily_volatility, bars_elapsed)
        return prob >= self.cfg.front_load_threshold


# ---------------------------------------------------------------------------
# IDX ARB-aware Almgren-Chriss planner
# ---------------------------------------------------------------------------

class IDXAlmgrenChrissPlanner:
    """
    ARB-aware execution planner.

    Normal mode (prob_hit_arb below threshold):
        Standard Almgren-Chriss hyperbolic-sine trajectory.
            w_t ∝ sinh(kappa * (T - t) / T)

    ARB-risk mode (prob_hit_arb >= threshold):
        Front-loaded exponential trajectory to guarantee liquidation before
        the LOB is locked.
            w_t ∝ exp(-lambda * t / T)
        With a very large lambda the first slice captures nearly the entire
        order quantity.

    Compatible drop-in for AlmgrenChrissPlanner in trading_engine.py.
    """

    def __init__(
        self,
        n_slices: int = 6,
        kappa: float = 1.0,
        monitor_cfg: Optional[ARBMonitorConfig] = None,
    ):
        self.n_slices = max(2, n_slices)
        self.kappa = max(0.1, kappa)
        self.monitor = ARBProbabilityMonitor(monitor_cfg or ARBMonitorConfig())

    def slice_delta(self, parent_delta: int) -> List[int]:
        """Standard Almgren-Chriss slice without ARB information."""
        return self._sinh_slice(parent_delta)

    def slice_delta_arb_aware(
        self,
        parent_delta: int,
        current_price: float,
        pred_daily_volatility: float,
        bars_elapsed: int = 0,
    ) -> List[int]:
        """
        Produce a child-order schedule that front-loads execution when the
        ARB probability is elevated.

        Args:
            parent_delta: Signed parent order quantity (positive = buy).
            current_price: Current IDX price in Rupiah.
            pred_daily_volatility: LSTM/Tobit predicted daily volatility.
            bars_elapsed: 1-minute bars elapsed in the current trading session.

        Returns:
            List of signed child quantities.
        """
        if parent_delta == 0:
            return []

        if self.monitor.is_arb_risk_elevated(current_price, pred_daily_volatility, bars_elapsed):
            return self._exponential_front_load(parent_delta)
        return self._sinh_slice(parent_delta)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sinh_slice(self, parent_delta: int) -> List[int]:
        """Hyperbolic-sine (standard Almgren-Chriss) schedule."""
        if parent_delta == 0:
            return []
        sign = 1 if parent_delta > 0 else -1
        qty = abs(parent_delta)
        n = self.n_slices
        raw = [math.sinh(self.kappa * (n - t) / n) for t in range(1, n + 1)]
        total = sum(raw)
        weights = [x / total for x in raw]
        chunks = [int(math.floor(qty * w)) for w in weights]
        rem = qty - sum(chunks)
        for i in range(rem):
            chunks[i % len(chunks)] += 1
        return [sign * c for c in chunks if c != 0]

    def _exponential_front_load(self, parent_delta: int) -> List[int]:
        """
        Exponential front-loaded schedule.
        w_t ∝ exp(-lambda * t / T)  for t = 1 .. n_slices
        A high lambda concentrates execution in the first slice.
        """
        if parent_delta == 0:
            return []
        sign = 1 if parent_delta > 0 else -1
        qty = abs(parent_delta)
        lam = self.monitor.cfg.front_load_lambda
        n = self.n_slices
        raw = [math.exp(-lam * t / n) for t in range(1, n + 1)]
        total = sum(raw)
        weights = [x / total for x in raw]
        chunks = [int(math.floor(qty * w)) for w in weights]
        rem = qty - sum(chunks)
        for i in range(rem):
            chunks[i % len(chunks)] += 1
        return [sign * c for c in chunks if c != 0]
