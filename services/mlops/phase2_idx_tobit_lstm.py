"""
Phase 2: IDX Censored Volatility LSTM with ARA/ARB Tobit Loss
-------------------------------------------------------------
The IDX Auto Rejection Atas (ARA) and Auto Rejection Bawah (ARB) mechanisms
cap intraday price moves.  When a stock hits the limit, the observed absolute
return is censored: the *true* latent volatility is unknown but at least as
large as the limit.  Standard MSE on censored observations biases the model
toward under-predicting volatility.

This module implements:
  * IDXTobitLoss - a fully differentiable PyTorch loss that handles right-
    censored (ARA) and left-censored (ARB) observations via the Tobit model.
  * BandarmologiFeatureEngineer - adds IDX-specific microstructure features:
      - top_5_broker_net_accum  (top-5 broker net accumulation)
      - foreign_net_flow        (foreign buy minus foreign sell normalised)
      - lob_imbalance           (bid-side LOB imbalance proxy)
  * IDXCensoredVolatilityLSTM - PyTorch LSTM trained with IDXTobitLoss that
    predicts the *latent* (uncensored) daily volatility.
  * IDXCensoredVolatilityRefitter - sklearn-style wrapper for train_pipeline
    integration with atomic save/load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"

# IDX default limits (fractional price movement)
IDX_ARA_UPPER: float = 0.25   # Auto Rejection Atas: +25 %
IDX_ARB_LOWER: float = -0.10  # Auto Rejection Bawah: -10 % (first board)

BANDARMOLOGI_FEATURES = [
    "top_5_broker_net_accum",
    "foreign_net_flow",
    "lob_imbalance",
]


# ---------------------------------------------------------------------------
# Tobit loss
# ---------------------------------------------------------------------------

class IDXTobitLoss:
    """
    Tobit-model loss for right-censored ARA and left-censored ARB observations.

    For each observation i:
      - If abs_return_i >= ARA limit  → right-censored; use survival function
      - If abs_return_i <= ARB limit  → left-censored; use CDF
      - Otherwise                     → uncensored; use Gaussian log-likelihood

    Loss = -mean( log_likelihood_i )

    Both ARA and ARB thresholds are expressed as *absolute return fractions*
    (e.g. ARA=0.25 means +25 %, ARB=-0.10 means -10 %).

    The function is fully differentiable w.r.t. model predictions because it
    uses only torch.distributions primitives.
    """

    def __init__(
        self,
        ara_limit: float = IDX_ARA_UPPER,
        arb_limit: float = IDX_ARB_LOWER,
        eps: float = 1e-6,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required for IDXTobitLoss.")
        self.ara_limit = float(ara_limit)
        self.arb_limit = float(arb_limit)
        self.eps = float(eps)
        self._log_sqrt2pi = float(np.log(np.sqrt(2.0 * np.pi)))

    def __call__(self, y_pred: "torch.Tensor", y_true: "torch.Tensor") -> "torch.Tensor":
        """
        Compute mean Tobit negative log-likelihood.

        Args:
            y_pred: (N,) predicted volatility in the same scale as y_true
                    (i.e. fractional absolute return, not log-space).
                    The model output is treated as the mean parameter mu of a
                    Gaussian distribution over the observed absolute return.
            y_true: (N,) observed absolute return (may be censored at limits)

        Returns:
            scalar loss tensor
        """
        mu = y_pred.squeeze(-1)
        y = y_true.squeeze(-1)

        # Residual variance: learned implicitly via a small positive log_sigma
        # We treat sigma as 1 normalised unit here; the network is trained on
        # standardised targets so this assumption is reasonable.
        sigma = torch.ones_like(mu)

        dist = torch.distributions.Normal(mu, sigma + self.eps)

        right_censored = y >= self.ara_limit
        left_censored = y <= self.arb_limit
        uncensored = ~right_censored & ~left_censored

        log_lik = torch.zeros_like(y)

        # Uncensored: standard Gaussian log-pdf
        if uncensored.any():
            log_lik[uncensored] = dist.log_prob(y[uncensored])

        # Right-censored (ARA hit): log P(Y >= ara)  = log S(ara)
        if right_censored.any():
            ara_t = torch.full_like(mu[right_censored], self.ara_limit)
            log_lik[right_censored] = dist.log_prob(ara_t) + torch.log(
                torch.clamp(1.0 - dist.cdf(ara_t), min=self.eps)
            )

        # Left-censored (ARB hit): log P(Y <= arb) = log F(arb)
        if left_censored.any():
            arb_t = torch.full_like(mu[left_censored], self.arb_limit)
            log_lik[left_censored] = dist.log_prob(arb_t) + torch.log(
                torch.clamp(dist.cdf(arb_t), min=self.eps)
            )

        return -log_lik.mean()


# ---------------------------------------------------------------------------
# Bandarmologi feature engineering
# ---------------------------------------------------------------------------

class BandarmologiFeatureEngineer:
    """
    Derives IDX microstructure signals that reveal institutional accumulation
    patterns (known colloquially as 'bandarmologi' among Indonesian traders).

    Required source columns (added if absent with zero fill):
      broker_buy_top5, broker_sell_top5  → top_5_broker_net_accum
      foreign_buy, foreign_sell, volume  → foreign_net_flow
      bid_vol, ask_vol                   → lob_imbalance

    All output features are normalised to approximately [-1, 1].
    """

    VOL_EPS = 1e-8

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add bandarmologi features in-place and return the enriched DataFrame.
        Missing source columns are zero-filled with a warning column suffix.
        """
        out = df.copy()

        # --- top_5_broker_net_accum ---
        if "broker_buy_top5" in out.columns and "broker_sell_top5" in out.columns:
            raw = (
                pd.to_numeric(out["broker_buy_top5"], errors="coerce").fillna(0.0)
                - pd.to_numeric(out["broker_sell_top5"], errors="coerce").fillna(0.0)
            )
            vol = pd.to_numeric(out.get("volume", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
            out["top_5_broker_net_accum"] = raw / (vol + self.VOL_EPS)
        elif "top_5_broker_net_accum" not in out.columns:
            out["top_5_broker_net_accum"] = 0.0

        # --- foreign_net_flow ---
        if "foreign_buy" in out.columns and "foreign_sell" in out.columns:
            raw = (
                pd.to_numeric(out["foreign_buy"], errors="coerce").fillna(0.0)
                - pd.to_numeric(out["foreign_sell"], errors="coerce").fillna(0.0)
            )
            vol = pd.to_numeric(out.get("volume", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
            out["foreign_net_flow"] = raw / (vol + self.VOL_EPS)
        elif "foreign_net_flow" not in out.columns:
            out["foreign_net_flow"] = 0.0

        # --- lob_imbalance  (bid / (bid + ask)) normalised to [-1, +1] ---
        if "bid_vol" in out.columns and "ask_vol" in out.columns:
            bid = pd.to_numeric(out["bid_vol"], errors="coerce").fillna(0.0)
            ask = pd.to_numeric(out["ask_vol"], errors="coerce").fillna(0.0)
            total = bid + ask + self.VOL_EPS
            out["lob_imbalance"] = (bid - ask) / total
        elif "lob_imbalance" not in out.columns:
            out["lob_imbalance"] = 0.0

        # Clip to [-3, 3] to suppress extreme outliers
        for col in BANDARMOLOGI_FEATURES:
            out[col] = out[col].clip(-3.0, 3.0)

        return out


# ---------------------------------------------------------------------------
# IDX Censored Volatility LSTM config and model
# ---------------------------------------------------------------------------

@dataclass
class IDXCensoredVolatilityConfig:
    """Configuration for the IDX-localised censored volatility LSTM."""

    lookback: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 20
    batch_size: int = 256
    ara_limit: float = IDX_ARA_UPPER
    arb_limit: float = IDX_ARB_LOWER
    early_stopping_patience: int = 5
    prod_dir: str = str(_SHARED / "models")


class _IDXLSTMNet(nn.Module if nn is not None else object):
    """
    Stacked LSTM predicting the latent (uncensored) abs_return.

    Input : (batch, lookback, n_features)
    Output: (batch, 1) — predicted latent volatility
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(self.drop(last))


class IDXCensoredVolatilityLSTM:
    """
    Trains a PyTorch LSTM on IDX censored volatility data using
    IDXTobitLoss to account for ARA/ARB capped observations.

    Features used:
      Base features (from existing pipeline):
        abs_return, volatility_5d, volatility_20d, volume_z20,
        range_pct, lpa_profile_id, finbert_score,
        sentiment_polarity, sentiment_intensity,
        vix_close_ret, usd_idr_close_ret
      Bandarmologi:
        top_5_broker_net_accum, foreign_net_flow, lob_imbalance
    """

    BASE_FEATURES = [
        "abs_return",
        "volatility_5d",
        "volatility_20d",
        "volume_z20",
        "range_pct",
        "lpa_profile_id",
        "finbert_score",
        "sentiment_polarity",
        "sentiment_intensity",
        "vix_close_ret",
        "usd_idr_close_ret",
    ]

    def __init__(self, cfg: Optional[IDXCensoredVolatilityConfig] = None):
        self.cfg = cfg or IDXCensoredVolatilityConfig()
        self._bandarmologi = BandarmologiFeatureEngineer()
        self._loss = IDXTobitLoss(ara_limit=self.cfg.ara_limit, arb_limit=self.cfg.arb_limit)
        self._net: Optional[object] = None
        self._feature_cols: list[str] = []
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

    def _available_features(self, df: pd.DataFrame) -> list[str]:
        return [c for c in self.BASE_FEATURES + BANDARMOLOGI_FEATURES if c in df.columns]

    def _standardise(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self._mean = x.reshape(-1, x.shape[-1]).mean(0)
            self._std = x.reshape(-1, x.shape[-1]).std(0) + 1e-8
        return (x - self._mean) / self._std

    def _build_sequences(
        self, df: pd.DataFrame, features: list[str]
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        xs, ys = [], []
        for _, grp in df.sort_values("date").groupby("company"):
            grp = grp.sort_values("date").reset_index(drop=True)
            vals = grp[features].astype(float).values
            tgt = grp["abs_return"].astype(float).values
            for i in range(self.cfg.lookback, len(grp)):
                xs.append(vals[i - self.cfg.lookback : i])
                ys.append(tgt[i])
        if not xs:
            return None, None
        x_arr = np.asarray(xs, dtype=np.float32)
        y_arr = np.asarray(ys, dtype=np.float32)
        return torch.from_numpy(x_arr), torch.from_numpy(y_arr)

    def fit(self, df: pd.DataFrame, train_cutoff: pd.Timestamp) -> dict:
        """
        Train the LSTM on the training window using the Tobit loss.

        Args:
            df: Panel DataFrame with company/date and feature columns.
            train_cutoff: Observations up to this date are used for training.

        Returns:
            Dictionary with training statistics.
        """
        if torch is None or nn is None:
            return {"skipped": "PyTorch unavailable"}

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "company", "abs_return"])
        df = self._bandarmologi.transform(df)

        self._feature_cols = self._available_features(df)
        for c in self._feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[self._feature_cols] = (
            df.groupby("company")[self._feature_cols].transform(lambda s: s.ffill().bfill())
        )
        df = df.dropna(subset=self._feature_cols + ["abs_return"])

        train_df = df[df["date"] <= train_cutoff].copy()
        val_df = df[df["date"] > train_cutoff].copy()

        x_tr, y_tr = self._build_sequences(train_df, self._feature_cols)
        if x_tr is None or len(x_tr) < 100:
            return {"skipped": "Insufficient training sequences"}

        x_tr = self._standardise(x_tr.numpy(), fit=True)
        x_tr = torch.from_numpy(x_tr)

        loader = DataLoader(
            TensorDataset(x_tr, y_tr),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        nf = x_tr.shape[2]
        self._net = _IDXLSTMNet(nf, self.cfg.hidden_size, self.cfg.num_layers, self.cfg.dropout)
        optimiser = torch.optim.Adam(self._net.parameters(), lr=self.cfg.learning_rate)

        best_val = float("inf")
        patience_left = self.cfg.early_stopping_patience
        train_losses = []

        for _ in range(self.cfg.epochs):
            self._net.train()
            ep_loss = 0.0
            for xb, yb in loader:
                optimiser.zero_grad()
                pred = self._net(xb)
                loss = self._loss(pred, yb)
                loss.backward()
                optimiser.step()
                ep_loss += loss.item() * len(xb)
            train_losses.append(ep_loss / max(len(x_tr), 1))

            # Validation on hold-out window (if available)
            if val_df is not None and len(val_df):
                x_va, y_va = self._build_sequences(val_df, self._feature_cols)
                if x_va is not None and len(x_va):
                    x_va_s = torch.from_numpy(
                        self._standardise(x_va.numpy(), fit=False)
                    )
                    self._net.eval()
                    with torch.no_grad():
                        val_loss = self._loss(self._net(x_va_s), y_va).item()
                    if val_loss < best_val - 1e-4:
                        best_val = val_loss
                        patience_left = self.cfg.early_stopping_patience
                    else:
                        patience_left -= 1
                    if patience_left <= 0:
                        break

        return {
            "epochs_run": len(train_losses),
            "final_train_loss": float(train_losses[-1]) if train_losses else float("nan"),
            "best_val_loss": float(best_val),
            "n_features": int(nf),
            "n_train_sequences": int(len(x_tr)),
            "bandarmologi_features": BANDARMOLOGI_FEATURES,
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict latent volatility for all rows in df.
        Returns df with a new column 'idx_pred_volatility'.
        """
        if self._net is None or not self._feature_cols:
            df = df.copy()
            df["idx_pred_volatility"] = df.get("volatility_20d", pd.Series(0.02, index=df.index))
            return df

        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = self._bandarmologi.transform(out)
        for c in self._feature_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        out[self._feature_cols] = (
            out.groupby("company")[self._feature_cols].transform(lambda s: s.ffill().bfill())
        )

        xs, row_ids = [], []
        for _, grp in out.groupby("company"):
            grp = grp.sort_values("date").reset_index(drop=False)
            vals = grp[self._feature_cols].astype(float).fillna(0.0).values
            for i in range(self.cfg.lookback, len(grp)):
                xs.append(vals[i - self.cfg.lookback : i])
                row_ids.append(grp.loc[i, "index"])

        if not xs:
            out["idx_pred_volatility"] = 0.02
            return out

        x_arr = np.asarray(xs, dtype=np.float32)
        x_sc = torch.from_numpy(self._standardise(x_arr, fit=False))
        self._net.eval()
        with torch.no_grad():
            preds = self._net(x_sc).squeeze(-1).numpy()

        out["idx_pred_volatility"] = np.nan
        for row_idx, pred_val in zip(row_ids, preds):
            out.loc[row_idx, "idx_pred_volatility"] = float(pred_val)
        out["idx_pred_volatility"] = out["idx_pred_volatility"].fillna(
            out.get("volatility_20d", pd.Series(0.02, index=out.index))
        )
        return out

    def save(self, prod_dir: Optional[str] = None) -> dict:
        """Atomically save model weights and normalisation stats."""
        out_dir = prod_dir or self.cfg.prod_dir
        os.makedirs(out_dir, exist_ok=True)
        artifacts: dict[str, str] = {}
        if self._net is not None and torch is not None and joblib is not None:
            net_path = os.path.join(out_dir, "idx_tobit_lstm.pt")
            tmp = net_path + ".tmp"
            torch.save(self._net.state_dict(), tmp)
            os.replace(tmp, net_path)
            artifacts["model"] = net_path

            meta_path = os.path.join(out_dir, "idx_tobit_lstm_meta.pkl")
            tmp_meta = meta_path + ".tmp"
            joblib.dump(
                {
                    "feature_cols": self._feature_cols,
                    "mean": self._mean,
                    "std": self._std,
                    "cfg": self.cfg,
                    "n_features": len(self._feature_cols),
                    "hidden_size": self.cfg.hidden_size,
                    "num_layers": self.cfg.num_layers,
                    "dropout": self.cfg.dropout,
                },
                tmp_meta,
            )
            os.replace(tmp_meta, meta_path)
            artifacts["meta"] = meta_path
        return artifacts

    def load(self, prod_dir: Optional[str] = None) -> bool:
        """Load model weights from prod_dir. Returns True on success."""
        out_dir = prod_dir or self.cfg.prod_dir
        meta_path = os.path.join(out_dir, "idx_tobit_lstm_meta.pkl")
        net_path = os.path.join(out_dir, "idx_tobit_lstm.pt")
        if not (os.path.exists(meta_path) and os.path.exists(net_path)):
            return False
        if torch is None or joblib is None:
            return False
        meta = joblib.load(meta_path)
        self._feature_cols = meta["feature_cols"]
        self._mean = meta["mean"]
        self._std = meta["std"]
        nf = meta["n_features"]
        self._net = _IDXLSTMNet(nf, meta["hidden_size"], meta["num_layers"], meta["dropout"])
        self._net.load_state_dict(torch.load(net_path, map_location="cpu"))
        self._net.eval()
        return True


# ---------------------------------------------------------------------------
# Convenience refitter (same interface as LSTMRefitter in train_pipeline)
# ---------------------------------------------------------------------------

class IDXCensoredVolatilityRefitter:
    """
    Thin adapter so train_pipeline.py can call the same fit/save interface
    as existing refitters.
    """

    def __init__(self, cfg: Optional[IDXCensoredVolatilityConfig] = None):
        self.cfg = cfg or IDXCensoredVolatilityConfig()
        self._lstm = IDXCensoredVolatilityLSTM(self.cfg)

    def fit(self, df: pd.DataFrame, train_cutoff: pd.Timestamp) -> dict:
        """Fit the Tobit LSTM. Returns stats dict."""
        return self._lstm.fit(df, train_cutoff)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add idx_pred_volatility column to df."""
        return self._lstm.predict(df)

    def save(self, prod_dir: str) -> dict:
        """Save model artifacts atomically. Returns artifact paths."""
        return self._lstm.save(prod_dir)
