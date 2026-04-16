"""
Phase 2: Global Panel LSTM with Entity Embeddings
--------------------------------------------------
Replaces per-ticker isolated LSTM loops with a single shared-weight network.
Entity embeddings learn latent asset representations for cross-asset
co-dependence and support transfer learning from liquid to micro-cap names.

Memory safety: sequences are generated lazily via tf.data generators so the
full market history is never loaded into RAM simultaneously.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import (
        Concatenate, Dense, Dropout, Embedding, Flatten, Input, LSTM,
    )
    from tensorflow.keras.models import Model
except Exception:  # pragma: no cover - optional at runtime
    tf = None
    Model = None

_SHARED = Path(__file__).resolve().parent.parent.parent / "shared"

PANEL_FEATURES = [
    "abs_return",
    "volatility_5d",
    "volatility_20d",
    "volume_z20",
    "range_pct",
    "finbert_score",
    "vix_close_ret",
    "usd_idr_close_ret",
    "lpa_profile_id",
]
MIN_TRAIN_SEQUENCES = 200


@dataclass
class PanelLSTMConfig:
    lookback: int = 10
    epochs: int = 30
    batch_size: int = 512
    embedding_dim: int = 16
    lstm_units_1: int = 64
    lstm_units_2: int = 32
    dense_units: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    early_stopping_patience: int = 7
    lr_patience: int = 4
    lr_factor: float = 0.5
    prod_dir: str = str(_SHARED / "models")
    extra_features: list = field(default_factory=list)


class GlobalPanelLSTMRefitter:
    """
    Entity Embedding + shared LSTM trained across the full asset universe.

    Architecture:
        ticker_id  -> Embedding(N+1, embedding_dim) -> Flatten
        sequence   -> LSTM(units_1) -> LSTM(units_2)
        Concatenate([lstm_out, embedding_flat])
        -> Dense(dense_units, relu) -> Dropout -> Dense(1)
    """

    def __init__(self, cfg: Optional[PanelLSTMConfig] = None):
        self.cfg = cfg or PanelLSTMConfig()
        self.ticker_map: dict[str, int] = {}
        self.scaler = StandardScaler()
        self.model: Optional[object] = None
        self._feature_cols: list[str] = []

    # ------------------------------------------------------------------
    # Ticker map — extensible for new IPOs
    # ------------------------------------------------------------------
    def build_ticker_map(self, companies: list[str]) -> dict[str, int]:
        """Assign stable integer IDs. New companies receive the next available ID."""
        max_id = max(self.ticker_map.values(), default=0)
        for company in sorted(set(companies)):
            if company not in self.ticker_map:
                max_id += 1
                self.ticker_map[company] = max_id
        return self.ticker_map

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build_model(self, n_features: int, n_tickers: int) -> object:
        if tf is None or Model is None:
            raise RuntimeError("TensorFlow is required for GlobalPanelLSTMRefitter.")
        cfg = self.cfg
        ticker_in = Input(shape=(1,), name="ticker_id", dtype="int32")
        emb = Embedding(
            input_dim=n_tickers + 1,
            output_dim=cfg.embedding_dim,
            name="entity_embedding",
        )(ticker_in)
        emb_flat = Flatten()(emb)

        seq_in = Input(shape=(cfg.lookback, n_features), name="market_sequence")
        x = LSTM(
            cfg.lstm_units_1,
            return_sequences=True,
            dropout=cfg.dropout_rate,
            recurrent_dropout=0.1,
        )(seq_in)
        x = LSTM(cfg.lstm_units_2, dropout=cfg.dropout_rate, recurrent_dropout=0.1)(x)

        merged = Concatenate()([x, emb_flat])
        x = Dense(cfg.dense_units, activation="relu")(merged)
        x = Dropout(cfg.dropout_rate)(x)
        output = Dense(1, name="vol_pred")(x)

        model = Model(inputs=[ticker_in, seq_in], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model

    # ------------------------------------------------------------------
    # Memory-safe sequence generator (never loads full history at once)
    # ------------------------------------------------------------------
    def _iter_sequences(self, df: pd.DataFrame, features: list[str], cutoff_date: Optional[pd.Timestamp]):
        """Lazily yields (ticker_id, sequence, target) for all companies."""
        for company, group in df.groupby("company"):
            tid = self.ticker_map.get(company, 0)
            g = group.sort_values("date").reset_index(drop=True)
            dates = pd.to_datetime(g["date"].values)
            vals = g[features].values.astype(np.float32)
            tgt = g["abs_return"].values.astype(np.float32)
            for i in range(self.cfg.lookback, len(g)):
                if cutoff_date is not None and dates[i] > cutoff_date:
                    continue
                yield (
                    np.int32(tid),
                    vals[i - self.cfg.lookback : i],
                    tgt[i],
                )

    def _count_sequences(self, df: pd.DataFrame, features: list[str], cutoff_date: Optional[pd.Timestamp]) -> int:
        """Lightweight count pass — iterates dates only, no numpy array allocation."""
        count = 0
        for _, group in df.groupby("company"):
            g = group.sort_values("date").reset_index(drop=True)
            dates = pd.to_datetime(g["date"].values)
            for i in range(self.cfg.lookback, len(g)):
                if cutoff_date is None or dates[i] <= cutoff_date:
                    count += 1
        return count

    def _make_dataset(
        self,
        df: pd.DataFrame,
        features: list[str],
        n_features: int,
        cutoff_date: Optional[pd.Timestamp],
    ) -> tuple[object, int]:
        """Build a memory-safe tf.data.Dataset via from_generator (truly lazy)."""
        if tf is None:
            raise RuntimeError("TensorFlow is required.")

        count = self._count_sequences(df, features, cutoff_date)
        if count == 0:
            return None, 0

        gen = lambda: self._iter_sequences(df, features, cutoff_date)  # noqa: E731
        raw_ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(self.cfg.lookback, n_features), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
            ),
        )
        ds = raw_ds.map(
            lambda tid, seq, y: (
                {"ticker_id": tf.expand_dims(tid, 0), "market_sequence": seq},
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.shuffle(min(count, 20_000), seed=42).batch(self.cfg.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds, count

    # ------------------------------------------------------------------
    # Scaler fitting (train split only — strict causal masking)
    # ------------------------------------------------------------------
    def _fit_scaler(self, df: pd.DataFrame, features: list[str], train_cutoff: pd.Timestamp) -> None:
        """Fit scaler on training rows only to prevent leakage."""
        train_rows = df[pd.to_datetime(df["date"]) <= train_cutoff]
        flat = train_rows[features].values.astype(np.float32)
        flat = flat[~np.isnan(flat).any(axis=1)]
        if len(flat) > 0:
            self.scaler.fit(flat)

    # ------------------------------------------------------------------
    # Preprocessing helper
    # ------------------------------------------------------------------
    def _prepare_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        features = [c for c in PANEL_FEATURES + self.cfg.extra_features if c in df.columns]
        work = df.copy()
        for c in features + ["abs_return"]:
            work[c] = pd.to_numeric(work[c], errors="coerce")
        work[features] = work.groupby("company")[features].transform(lambda s: s.ffill().bfill())
        work[features] = pd.DataFrame(
            self.scaler.transform(work[features].fillna(0.0)),
            columns=features,
            index=work.index,
        )
        work = work.dropna(subset=features + ["abs_return"])
        return work, features

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, train_cutoff: pd.Timestamp) -> dict:
        if tf is None:
            raise RuntimeError("TensorFlow is required for GlobalPanelLSTMRefitter.")

        raw_features = [c for c in PANEL_FEATURES + self.cfg.extra_features if c in df.columns]
        companies = sorted(df["company"].dropna().unique().tolist())
        self.build_ticker_map(companies)
        n_tickers = max(self.ticker_map.values(), default=1)

        self._fit_scaler(df, raw_features, train_cutoff)
        work, features = self._prepare_df(df)
        self._feature_cols = features
        n_features = len(features)

        val_cutoff = train_cutoff + pd.Timedelta(days=1)
        max_date = pd.to_datetime(df["date"]).max()

        train_ds, n_train = self._make_dataset(work, features, n_features, train_cutoff)
        val_ds, n_val = self._make_dataset(
            work,
            features,
            n_features,
            max_date,
        )
        # Re-create val from rows after train_cutoff for proper validation
        val_work = work[pd.to_datetime(work["date"]) > train_cutoff]
        val_ds, n_val = self._make_dataset(val_work, features, n_features, None)

        if n_train < MIN_TRAIN_SEQUENCES:
            raise RuntimeError(
                f"Insufficient training sequences ({n_train}) for GlobalPanelLSTMRefitter."
            )

        self.model = self._build_model(n_features, n_tickers)
        callbacks = [
            EarlyStopping(
                monitor="val_loss" if n_val > 0 else "loss",
                patience=self.cfg.early_stopping_patience,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss" if n_val > 0 else "loss",
                factor=self.cfg.lr_factor,
                patience=self.cfg.lr_patience,
                min_lr=1e-6,
            ),
        ]
        fit_kwargs = {"epochs": self.cfg.epochs, "callbacks": callbacks, "verbose": 0}
        if n_val > 0:
            fit_kwargs["validation_data"] = val_ds

        hist = self.model.fit(train_ds, **fit_kwargs)
        epochs_run = int(len(hist.history["loss"]))
        val_loss = float(hist.history.get("val_loss", [float("nan")])[-1])

        return {
            "n_train_sequences": n_train,
            "n_val_sequences": n_val,
            "epochs_run": epochs_run,
            "final_val_loss": val_loss,
            "n_tickers": n_tickers,
            "embedding_dim": self.cfg.embedding_dim,
            "feature_cols": features,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, prod_dir: Optional[str] = None) -> dict:
        out_dir = prod_dir or self.cfg.prod_dir
        os.makedirs(out_dir, exist_ok=True)
        artifacts = {}

        if self.model is not None:
            model_path = os.path.join(out_dir, "global_panel_lstm.keras")
            tmp = model_path + ".tmp"
            self.model.save(tmp)
            os.replace(tmp, model_path)
            artifacts["model"] = model_path

        meta_path = os.path.join(out_dir, "global_panel_lstm_meta.pkl")
        tmp = meta_path + ".tmp"
        joblib.dump(
            {
                "ticker_map": self.ticker_map,
                "scaler": self.scaler,
                "feature_cols": self._feature_cols,
                "cfg": self.cfg,
            },
            tmp,
        )
        os.replace(tmp, meta_path)
        artifacts["meta"] = meta_path
        return artifacts

    def load(self, prod_dir: Optional[str] = None) -> None:
        out_dir = prod_dir or self.cfg.prod_dir
        meta_path = os.path.join(out_dir, "global_panel_lstm_meta.pkl")
        if os.path.exists(meta_path):
            saved = joblib.load(meta_path)
            self.ticker_map = saved.get("ticker_map", {})
            self.scaler = saved.get("scaler", StandardScaler())
            self._feature_cols = saved.get("feature_cols", [])

        model_path = os.path.join(out_dir, "global_panel_lstm.keras")
        if os.path.exists(model_path) and tf is not None:
            self.model = tf.keras.models.load_model(model_path)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Train Global Panel LSTM with entity embeddings")
    p.add_argument("--data", default=str(_SHARED / "data" / "phase6_lpa_enriched.csv"))
    p.add_argument("--prod-dir", default=str(_SHARED / "models"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lookback", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "company", "abs_return"]).sort_values(["company", "date"])

    max_date = df["date"].max()
    train_cutoff = max_date - pd.Timedelta(days=45)
    cfg = PanelLSTMConfig(epochs=args.epochs, lookback=args.lookback, batch_size=args.batch_size, prod_dir=args.prod_dir)
    refitter = GlobalPanelLSTMRefitter(cfg)
    stats = refitter.fit(df, train_cutoff=train_cutoff)
    artifacts = refitter.save()
    print(f"STATS={stats}")
    print(f"ARTIFACTS={artifacts}")


if __name__ == "__main__":
    main()
