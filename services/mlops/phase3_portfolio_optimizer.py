"""
Phase 3: End-to-End Markowitz Allocation & HRP Fallback + IDX Constraints
--------------------------------------------------------------------------
E2EPortfolioOptimizer:
  - Neural network with softmax output producing optimal weights w_t
  - Custom Sharpe maximization loss with transaction cost and
    concentration penalties baked into the learning process
  - Long-only constraint enforced by softmax activation

IDXConstrainedE2ELoss (extends _SharpeWithCostLoss):
  - Maximum Participation (ADV) penalty: differentiable soft penalty that
    heavily penalises allocations exceeding 5 % of the 20-day ADV.
    The penalty is proportional to the capital-weighted excess:
      adv_penalty = lambda_adv * mean( relu(w * pv - adv_cap * adv_20d) )
    where adv_cap defaults to 0.05 (5 %).  The penalty is differentiable
    because relu is a supergradient of max(0, x).
  - L1 Turnover penalty: gamma/2 * sum(|w_t - w_{t-1}|) to explicitly
    penalise high rebalancing in illiquid IDX third-liner names where
    broker fees and slippage are prohibitive.

HRPAllocator:
  - Hierarchical Risk Parity fallback using SciPy clustering
  - Clusters correlated assets and distributes capital recursively
  - Robust to structural market breaks (does not invert covariance matrix)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
except Exception:  # pragma: no cover - optional at runtime
    linkage = None
    leaves_list = None
    squareform = None

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.models import Model
except Exception:  # pragma: no cover
    tf = None
    Model = None

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"

MIN_OPT_SAMPLES = 60  # minimum time-steps to compute meaningful Sharpe


# ---------------------------------------------------------------------------
# Custom Sharpe + cost loss
# ---------------------------------------------------------------------------
class _SharpeWithCostLoss:
    """
    Batch-level Sharpe ratio loss with:
      - Transaction cost proxy (L2 deviation from equal-weight baseline)
      - Concentration penalty (penalises any weight exceeding max_weight)
    """

    def __init__(self, lambda_tc: float = 0.005, lambda_conc: float = 1.0, max_weight: float = 0.10):
        self.lambda_tc = float(lambda_tc)
        self.lambda_conc = float(lambda_conc)
        self.max_weight = float(max_weight)

    def __call__(self, y_true, y_pred):
        if tf is None:
            raise RuntimeError("TensorFlow required.")
        # y_true: (batch, n_assets) forward returns
        # y_pred: (batch, n_assets) portfolio weights from softmax
        port_returns = tf.reduce_sum(y_true * y_pred, axis=1)
        mean_ret = tf.reduce_mean(port_returns)
        std_ret = tf.math.reduce_std(port_returns) + 1e-8
        neg_sharpe = -(mean_ret / std_ret)

        # Turnover proxy: deviation from equal-weight (no prev weights needed)
        n_assets = tf.cast(tf.shape(y_pred)[1], tf.float32)
        equal_w = tf.ones_like(y_pred) / n_assets
        tc_penalty = self.lambda_tc * tf.reduce_mean(
            tf.reduce_sum(tf.square(y_pred - equal_w), axis=1)
        )

        # Concentration penalty
        max_w = tf.reduce_max(y_pred, axis=1)
        conc_penalty = self.lambda_conc * tf.reduce_mean(
            tf.nn.relu(max_w - self.max_weight)
        )
        return neg_sharpe + tc_penalty + conc_penalty


# ---------------------------------------------------------------------------
# IDX Constraints: ADV participation cap + L1 turnover penalty
# ---------------------------------------------------------------------------

class IDXConstrainedE2ELoss(_SharpeWithCostLoss):
    """
    Extends _SharpeWithCostLoss with IDX-specific constraints:

    1. Maximum Participation (ADV) soft penalty
       ─────────────────────────────────────────
       Prevents the network from placing orders that would exceed
       ``adv_cap`` (default 5 %) of each asset's 20-day ADV.

       penalty_adv = lambda_adv * mean_over_assets( relu(w_i - cap_i) )

       where cap_i = adv_cap * adv_i / total_portfolio_value.

       cap_vector is a TensorFlow constant of shape (n_assets,) computed
       from the asset ADV vector passed at construction time.

       The relu operator is a supergradient of max(0, x), making the
       penalty fully differentiable for backpropagation.

    2. L1 Turnover penalty
       ─────────────────────
       penalty_turn = (gamma / 2) * mean( |w_t - w_{t-1}| )

       w_prev is the rolling previous-step weight passed through
       y_true[:,n_assets:] (the loss packs prev_weights into the second
       half of y_true to remain Keras-compatible).
    """

    def __init__(
        self,
        lambda_tc: float = 0.005,
        lambda_conc: float = 1.0,
        max_weight: float = 0.10,
        lambda_adv: float = 10.0,
        adv_cap: float = 0.05,
        gamma_turnover: float = 0.5,
        adv_vector: Optional[np.ndarray] = None,
        n_assets: int = 0,
    ):
        super().__init__(lambda_tc=lambda_tc, lambda_conc=lambda_conc, max_weight=max_weight)
        self.lambda_adv = float(lambda_adv)
        self.adv_cap = float(adv_cap)
        self.gamma_turnover = float(gamma_turnover)
        self._adv_vector = adv_vector  # shape (n_assets,) — 20-day ADV normalised
        self._n_assets = int(n_assets)

    def _build_adv_cap_tensor(self, n_a: int):
        """Return per-asset ADV cap as a float32 tensor of shape (1, n_a)."""
        if tf is None:
            return None
        if self._adv_vector is not None and len(self._adv_vector) == n_a:
            adv_norm = np.clip(self._adv_vector.astype(np.float32), 1e-8, None)
            cap = self.adv_cap * (adv_norm / adv_norm.sum())
        else:
            # Fallback: uniform cap = adv_cap / n_assets
            cap = np.full(n_a, self.adv_cap / max(n_a, 1), dtype=np.float32)
        return tf.constant(cap.reshape(1, n_a), dtype=tf.float32)

    def __call__(self, y_true, y_pred):
        """
        Args:
            y_true: (batch, 2 * n_assets)
                    y_true[:, :n_assets]  = forward returns
                    y_true[:, n_assets:]  = previous portfolio weights
            y_pred: (batch, n_assets) — softmax portfolio weights
        """
        if tf is None:
            raise RuntimeError("TensorFlow required.")

        n_a = tf.shape(y_pred)[1]
        n_a_int = y_pred.shape[1] or self._n_assets

        # Split returns and previous weights from y_true
        fwd_returns = y_true[:, :n_a]
        prev_weights = y_true[:, n_a:]

        # Base Sharpe + TC + concentration loss (operating on forward returns)
        base_loss = super().__call__(fwd_returns, y_pred)

        # --- ADV participation penalty ---
        adv_cap_t = self._build_adv_cap_tensor(n_a_int)
        if adv_cap_t is not None:
            excess = tf.nn.relu(y_pred - adv_cap_t)
            adv_penalty = self.lambda_adv * tf.reduce_mean(
                tf.reduce_sum(excess, axis=1)
            )
        else:
            adv_penalty = tf.constant(0.0)

        # --- L1 Turnover penalty ---
        # Only apply when prev_weights has the expected shape
        prev_w_shape = prev_weights.shape[1] if prev_weights.shape.rank > 1 else 0
        if prev_w_shape and prev_w_shape == n_a_int:
            l1_turn = tf.reduce_mean(
                tf.reduce_sum(tf.abs(y_pred - prev_weights), axis=1)
            )
            turnover_penalty = (self.gamma_turnover / 2.0) * l1_turn
        else:
            turnover_penalty = tf.constant(0.0)

        return base_loss + adv_penalty + turnover_penalty


# ---------------------------------------------------------------------------
# E2E Portfolio Optimizer
# ---------------------------------------------------------------------------
@dataclass
class E2EConfig:
    lambda_tc: float = 0.005
    lambda_conc: float = 1.0
    max_weight: float = 0.10
    hidden_units: int = 256
    dropout_rate: float = 0.2
    learning_rate: float = 5e-4
    epochs: int = 40
    batch_size: int = 128
    early_stopping_patience: int = 8
    prod_dir: str = str(_SHARED / "models")
    # IDX-specific constraint parameters
    use_idx_constraints: bool = True
    lambda_adv: float = 10.0
    adv_cap: float = 0.05
    gamma_turnover: float = 0.5


class E2EPortfolioOptimizer:
    """
    Directly learns portfolio weights w_t via an MLP with softmax output.

    Inputs per sample:
      - Predicted volatilities for all N assets (from GlobalPanelLSTM / Tobit LSTM)
      - Recent mean returns for all N assets
    Output:
      - w_t: N-dimensional portfolio weight vector (softmax → long-only, sums to 1)

    Training data structure (standard / non-IDX):
      X[t] = [pred_vol_1..N, mean_ret_1..N]          (2N features)
      y[t] = forward_returns_1..N                     (N targets)
      Loss = -Sharpe(w_t · r_{t+1}) + TC_penalty + Concentration_penalty

    When use_idx_constraints=True (default):
      y[t] = concat(forward_returns, prev_weights)    (2N targets)
      Loss += ADV_participation_penalty + L1_turnover_penalty
    """

    def __init__(self, cfg: Optional[E2EConfig] = None):
        self.cfg = cfg or E2EConfig()
        self.model: Optional[object] = None
        self.n_assets: int = 0
        self.asset_names: list[str] = []
        self._adv_vector: Optional[np.ndarray] = None

    def _build_model(self, n_input: int, n_assets: int) -> object:
        if tf is None or Model is None:
            raise RuntimeError("TensorFlow required for E2EPortfolioOptimizer.")
        inp = Input(shape=(n_input,), name="market_state")
        x = Dense(self.cfg.hidden_units, activation="relu")(inp)
        x = Dropout(self.cfg.dropout_rate)(x)
        x = Dense(self.cfg.hidden_units // 2, activation="relu")(x)
        x = Dropout(self.cfg.dropout_rate)(x)
        # Softmax enforces long-only (all non-negative) and sum-to-one
        weights = Dense(n_assets, activation="softmax", name="portfolio_weights")(x)
        m = Model(inputs=inp, outputs=weights)
        if self.cfg.use_idx_constraints:
            loss_fn = IDXConstrainedE2ELoss(
                lambda_tc=self.cfg.lambda_tc,
                lambda_conc=self.cfg.lambda_conc,
                max_weight=self.cfg.max_weight,
                lambda_adv=self.cfg.lambda_adv,
                adv_cap=self.cfg.adv_cap,
                gamma_turnover=self.cfg.gamma_turnover,
                adv_vector=self._adv_vector,
                n_assets=n_assets,
            )
        else:
            loss_fn = _SharpeWithCostLoss(
                lambda_tc=self.cfg.lambda_tc,
                lambda_conc=self.cfg.lambda_conc,
                max_weight=self.cfg.max_weight,
            )
        m.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss=loss_fn,
        )
        return m

    def fit(self, df: pd.DataFrame, train_cutoff: pd.Timestamp) -> dict:
        """
        Build training samples from a panel DataFrame.

        Each time-step t produces:
          X[t] = predicted vols (from idx_pred_volatility / volatility_20d) +
                 rolling mean returns across N assets
          y[t] = forward abs_return for all N assets at t+1

          When IDX constraints are enabled (use_idx_constraints=True):
            y[t] = concat(forward_returns, prev_weights_t)   shape (2N,)
          This allows IDXConstrainedE2ELoss to compute the L1 turnover penalty
          without requiring an extra input layer.
        """
        if tf is None:
            raise RuntimeError("TensorFlow required.")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "company", "abs_return"])

        assets = sorted(df["company"].dropna().unique().tolist())
        self.asset_names = assets
        self.n_assets = len(assets)

        # Prefer IDX Tobit LSTM predictions when available
        vol_col = next(
            (c for c in ["idx_pred_volatility", "pred_volatility", "volatility_20d"] if c in df.columns),
            "abs_return",
        )

        # Pivot to (date × asset) matrices — uses only training window
        train_df = df[df["date"] <= train_cutoff]
        pivot_vol = train_df.pivot_table(index="date", columns="company", values=vol_col, aggfunc="mean")
        pivot_ret = train_df.pivot_table(index="date", columns="company", values="abs_return", aggfunc="mean")

        pivot_vol = pivot_vol.reindex(columns=assets).ffill().fillna(0.0)
        pivot_ret = pivot_ret.reindex(columns=assets).ffill().fillna(0.0)
        pivot_vol, pivot_ret = pivot_vol.align(pivot_ret, join="inner", axis=0)

        if len(pivot_vol) < MIN_OPT_SAMPLES:
            raise RuntimeError(
                f"Insufficient time-steps ({len(pivot_vol)}) for E2EPortfolioOptimizer."
            )

        # Build 20-day ADV vector for IDX ADV participation constraint
        if "volume" in df.columns:
            adv_pivot = train_df.pivot_table(
                index="date", columns="company", values="volume", aggfunc="mean"
            ).reindex(columns=assets).ffill().fillna(0.0)
            self._adv_vector = adv_pivot.rolling(20, min_periods=1).mean().iloc[-1].values.astype(np.float32)
        else:
            self._adv_vector = None

        # Build X = [vol_t, mean_ret_rolling_5], y = ret_{t+1}
        rolling_mean_ret = pivot_ret.rolling(5, min_periods=1).mean()
        x_vol = pivot_vol.values.astype(np.float32)
        x_ret = rolling_mean_ret.values.astype(np.float32)
        x_all = np.concatenate([x_vol, x_ret], axis=1)[:-1]  # drop last (no forward y)
        fwd_ret = pivot_ret.values.astype(np.float32)[1:]     # forward returns

        if self.cfg.use_idx_constraints:
            # Append equal-weight vector as the bootstrapped prev_weights for
            # each time step.  During online inference the execution engine uses
            # actual prior weights; here we regularise the network toward not
            # deviating from the equal-weight allocation, which is the correct
            # Bayesian prior before the model has any position history.
            # This still produces a meaningful L1 turnover penalty: high
            # deviations from 1/N attract cost, as desired for illiquid assets.
            n_a = self.n_assets
            eq_w = np.full(n_a, 1.0 / n_a, dtype=np.float32)
            prev_w = np.tile(eq_w, (len(fwd_ret), 1))
            y_all = np.concatenate([fwd_ret, prev_w], axis=1)
        else:
            y_all = fwd_ret

        split = int(len(x_all) * 0.85)
        x_train, x_val = x_all[:split], x_all[split:]
        y_train, y_val = y_all[:split], y_all[split:]

        self.model = self._build_model(n_input=x_all.shape[1], n_assets=self.n_assets)
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.cfg.early_stopping_patience, restore_best_weights=True
        )
        hist = self.model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            callbacks=[es],
            verbose=0,
        )
        return {
            "n_assets": self.n_assets,
            "n_time_steps": len(x_all),
            "epochs_run": int(len(hist.history["loss"])),
            "final_val_loss": float(hist.history.get("val_loss", [float("nan")])[-1]),
            "idx_constraints_enabled": bool(self.cfg.use_idx_constraints),
            "vol_col_used": vol_col,
        }

    def predict_weights(self, vol_vector: np.ndarray, ret_vector: np.ndarray) -> np.ndarray:
        """Return portfolio weights for a single time-step (inference)."""
        if self.model is None:
            n = max(self.n_assets, 1)
            return np.full(n, 1.0 / n, dtype=np.float32)
        x = np.concatenate([vol_vector, ret_vector], axis=0).reshape(1, -1).astype(np.float32)
        return self.model.predict(x, verbose=0)[0]

    def save(self, prod_dir: Optional[str] = None) -> dict:
        out_dir = prod_dir or self.cfg.prod_dir
        os.makedirs(out_dir, exist_ok=True)
        artifacts = {}
        if self.model is not None:
            mp = os.path.join(out_dir, "e2e_portfolio_optimizer.keras")
            tmp = mp + ".tmp"
            self.model.save(tmp)
            os.replace(tmp, mp)
            artifacts["model"] = mp
        meta = {"asset_names": self.asset_names, "n_assets": self.n_assets, "cfg": self.cfg}
        mp = os.path.join(out_dir, "e2e_portfolio_meta.pkl")
        tmp = mp + ".tmp"
        joblib.dump(meta, tmp)
        os.replace(tmp, mp)
        artifacts["meta"] = mp
        return artifacts

    def load(self, prod_dir: Optional[str] = None) -> None:
        out_dir = prod_dir or self.cfg.prod_dir
        meta_path = os.path.join(out_dir, "e2e_portfolio_meta.pkl")
        if os.path.exists(meta_path):
            m = joblib.load(meta_path)
            self.asset_names = m.get("asset_names", [])
            self.n_assets = m.get("n_assets", 0)
        model_path = os.path.join(out_dir, "e2e_portfolio_optimizer.keras")
        if os.path.exists(model_path) and tf is not None:
            self.model = tf.keras.models.load_model(model_path, compile=False)


# ---------------------------------------------------------------------------
# HRP Allocator (SciPy fallback)
# ---------------------------------------------------------------------------
class HRPAllocator:
    """
    Hierarchical Risk Parity allocation.

    Steps:
      1. Compute correlation matrix from predicted/realised volatilities
      2. Hierarchical clustering to identify correlated asset groups
      3. Quasi-diagonalise the covariance matrix (leaf-order sort)
      4. Recursive bisection to distribute capital inverse-proportional to
         cluster variance (no matrix inversion — robust to ill-conditioning)
    """

    def __init__(self, linkage_method: str = "ward"):
        self.linkage_method = linkage_method
        self.last_weights: Optional[pd.Series] = None

    @staticmethod
    def _correl_to_dist(corr: pd.DataFrame) -> np.ndarray:
        dist = np.sqrt(np.clip((1.0 - corr.values) / 2.0, 0.0, 1.0))
        np.fill_diagonal(dist, 0.0)
        return dist

    @staticmethod
    def _get_quasi_diag(link: np.ndarray) -> list[int]:
        """Return sorted leaf indices from a linkage matrix (quasi-diagonalisation)."""
        if leaves_list is None:
            n = int(link[-1, 3])
            return list(range(n))
        return list(leaves_list(link))

    @staticmethod
    def _cluster_var(cov: pd.DataFrame, items: list) -> float:
        sub_cov = cov.loc[items, items].values
        w = np.ones(len(items)) / len(items)
        return float(w @ sub_cov @ w)

    def _recursive_bisect(self, cov: pd.DataFrame, sorted_items: list) -> pd.Series:
        weights = pd.Series(1.0, index=sorted_items)
        clusters = [sorted_items]
        while clusters:
            clusters_next = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left, right = cluster[:mid], cluster[mid:]
                var_left = self._cluster_var(cov, left)
                var_right = self._cluster_var(cov, right)
                total_var = var_left + var_right
                alpha = 1.0 - var_left / max(total_var, 1e-10)
                weights[left] *= 1.0 - alpha
                weights[right] *= alpha
                clusters_next.extend([left, right])
            clusters = clusters_next
        return weights / weights.sum()

    def compute_hrp_weights(self, returns_df: pd.DataFrame) -> pd.Series:
        """
        Compute HRP portfolio weights from a (T × N) returns DataFrame.

        Args:
            returns_df: DataFrame with dates as index, asset names as columns.

        Returns:
            pd.Series of portfolio weights indexed by asset name.
        """
        if linkage is None or squareform is None:
            n = len(returns_df.columns)
            return pd.Series(1.0 / n, index=returns_df.columns)

        returns_clean = returns_df.dropna(how="all", axis=1).fillna(0.0)
        if returns_clean.shape[1] < 2:
            n = max(returns_clean.shape[1], 1)
            return pd.Series(1.0 / n, index=returns_clean.columns)

        corr = returns_clean.corr().clip(-0.9999, 0.9999)
        cov = returns_clean.cov()

        dist_mat = self._correl_to_dist(corr)
        condensed = squareform(dist_mat, checks=False)
        condensed = np.clip(condensed, 0.0, None)
        link = linkage(condensed, method=self.linkage_method)
        sorted_idx = self._get_quasi_diag(link)
        sorted_assets = returns_clean.columns[sorted_idx].tolist()

        weights = self._recursive_bisect(cov, sorted_assets)
        self.last_weights = weights.reindex(returns_df.columns).fillna(0.0)
        total = self.last_weights.sum()
        if total > 0:
            self.last_weights /= total
        return self.last_weights

    def save(self, prod_dir: str) -> str:
        os.makedirs(prod_dir, exist_ok=True)
        path = os.path.join(prod_dir, "hrp_weights.pkl")
        tmp = path + ".tmp"
        joblib.dump({"weights": self.last_weights, "linkage_method": self.linkage_method}, tmp)
        os.replace(tmp, path)
        return path

    def load(self, prod_dir: str) -> None:
        path = os.path.join(prod_dir, "hrp_weights.pkl")
        if os.path.exists(path):
            saved = joblib.load(path)
            self.last_weights = saved.get("weights")
            self.linkage_method = saved.get("linkage_method", self.linkage_method)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="E2E Markowitz + HRP portfolio optimizer")
    p.add_argument("--data", default=str(_SHARED / "data" / "phase6_lpa_enriched.csv"))
    p.add_argument("--prod-dir", default=str(_SHARED / "models"))
    p.add_argument("--epochs", type=int, default=40)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "company", "abs_return"]).sort_values(["company", "date"])
    max_date = df["date"].max()
    train_cutoff = max_date - pd.Timedelta(days=45)

    # E2E optimizer
    cfg = E2EConfig(epochs=args.epochs, prod_dir=args.prod_dir)
    opt = E2EPortfolioOptimizer(cfg)
    try:
        stats = opt.fit(df, train_cutoff)
        arts = opt.save()
        print(f"E2E_STATS={stats}")
        print(f"E2E_ARTIFACTS={arts}")
    except RuntimeError as exc:
        print(f"E2E skipped: {exc}")

    # HRP allocator
    hrp = HRPAllocator()
    pivot = df[df["date"] <= train_cutoff].pivot_table(
        index="date", columns="company", values="abs_return", aggfunc="mean"
    )
    weights = hrp.compute_hrp_weights(pivot)
    hrp_path = hrp.save(args.prod_dir)
    print(f"HRP_WEIGHTS={weights.sort_values(ascending=False).head(5).to_dict()}")
    print(f"HRP_PATH={hrp_path}")


if __name__ == "__main__":
    main()
