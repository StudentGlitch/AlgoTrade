"""
Phase 3: End-to-End Markowitz Allocation & HRP Fallback
-------------------------------------------------------
E2EPortfolioOptimizer:
  - Neural network with softmax output producing optimal weights w_t
  - Custom Sharpe maximization loss with transaction cost and
    concentration penalties baked into the learning process
  - Long-only constraint enforced by softmax activation

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


class E2EPortfolioOptimizer:
    """
    Directly learns portfolio weights w_t via an MLP with softmax output.

    Inputs per sample:
      - Predicted volatilities for all N assets (from GlobalPanelLSTM)
      - Recent mean returns for all N assets
    Output:
      - w_t: N-dimensional portfolio weight vector (softmax → long-only, sums to 1)

    Training data structure:
      X[t] = [pred_vol_1..N, mean_ret_1..N]  (2N features)
      y[t] = forward_returns_1..N             (N targets)
    Loss = -Sharpe(w_t · r_{t+1}) + TC_penalty + Concentration_penalty
    """

    def __init__(self, cfg: Optional[E2EConfig] = None):
        self.cfg = cfg or E2EConfig()
        self.model: Optional[object] = None
        self.n_assets: int = 0
        self.asset_names: list[str] = []

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
          X[t] = predicted vols (from pred_volatility or volatility_20d) +
                 rolling mean returns across N assets
          y[t] = forward abs_return for all N assets at t+1
        """
        if tf is None:
            raise RuntimeError("TensorFlow required.")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "company", "abs_return"])

        assets = sorted(df["company"].dropna().unique().tolist())
        self.asset_names = assets
        self.n_assets = len(assets)

        vol_col = "pred_volatility" if "pred_volatility" in df.columns else "volatility_20d"
        if vol_col not in df.columns:
            vol_col = "abs_return"

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

        # Build X = [vol_t, mean_ret_rolling_5], y = ret_{t+1}
        rolling_mean_ret = pivot_ret.rolling(5, min_periods=1).mean()
        X_vol = pivot_vol.values.astype(np.float32)
        X_ret = rolling_mean_ret.values.astype(np.float32)
        X = np.concatenate([X_vol, X_ret], axis=1)[:-1]  # drop last (no forward y)
        y = pivot_ret.values.astype(np.float32)[1:]       # forward returns

        split = int(len(X) * 0.85)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.model = self._build_model(n_input=X.shape[1], n_assets=self.n_assets)
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.cfg.early_stopping_patience, restore_best_weights=True
        )
        hist = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            callbacks=[es],
            verbose=0,
        )
        return {
            "n_assets": self.n_assets,
            "n_time_steps": len(X),
            "epochs_run": int(len(hist.history["loss"])),
            "final_val_loss": float(hist.history.get("val_loss", [float("nan")])[-1]),
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
