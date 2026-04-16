"""
Phase 1: IDX Localized NLP and Pom-Pom Regime Detection
--------------------------------------------------------
Provides Indonesian-language sentiment extraction using a localized
HuggingFace model and a GMM-based Pom-Pom regime detector that identifies
pump-and-dump retail activity on the IDX by combining localized sentiment
dimensions with abnormal volume spikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


@dataclass
class LocalizedSentimentConfig:
    model_name: str = "indobenchmark/indobert-base-p1"
    batch_size: int = 16
    max_length: int = 256


class LocalizedIndonesianSentimentExtractor:
    """
    Extract localized IDX sentiment dimensions:
    - polarity in [-1, 1]
    - intensity in [0, 1]
    - uncertainty in [0, 1]
    """

    def __init__(self, cfg: Optional[LocalizedSentimentConfig] = None):
        self.cfg = cfg or LocalizedSentimentConfig()
        self._model_ref = None
        self._positive_idx = 0
        self._negative_idx = 1

    def _load_model(self):
        if self._model_ref is not None:
            return self._model_ref
        if AutoTokenizer is None or AutoModelForSequenceClassification is None or torch is None:
            self._model_ref = None
            return None
        tok = AutoTokenizer.from_pretrained(self.cfg.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.cfg.model_name)
        model.eval()
        id2label = getattr(model.config, "id2label", {}) or {}
        pos_idx = None
        neg_idx = None
        for idx, label in id2label.items():
            txt = str(label).strip().lower()
            if txt in {"positive", "pos", "bullish"}:
                pos_idx = int(idx)
            if txt in {"negative", "neg", "bearish"}:
                neg_idx = int(idx)
        if pos_idx is not None:
            self._positive_idx = pos_idx
        if neg_idx is not None:
            self._negative_idx = neg_idx
        self._model_ref = (tok, model)
        return self._model_ref

    def _to_dimensions_from_probs(self, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_labels = probs.shape[1]
        if n_labels >= 2:
            # Uses model.config.id2label (if present) to map positive/negative
            # indices; otherwise falls back to {positive:0, negative:1}.
            pos_idx = self._positive_idx if self._positive_idx < n_labels else 0
            neg_idx = self._negative_idx if self._negative_idx < n_labels else 1
            positive = probs[:, pos_idx]
            negative = probs[:, neg_idx]
        else:
            positive = probs[:, 0]
            negative = np.zeros_like(positive)

        polarity = np.clip(positive - negative, -1.0, 1.0)
        intensity = probs.max(axis=1)
        entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
        uncertainty = np.clip(entropy / np.log(max(n_labels, 2)), 0.0, 1.0)
        return polarity, intensity, uncertainty

    def score_texts(self, texts: list[str]) -> pd.DataFrame:
        if not texts:
            return pd.DataFrame(
                columns=["sentiment_polarity", "sentiment_intensity", "sentiment_uncertainty"]
            )

        model_ref = self._load_model()
        if model_ref is None:
            return pd.DataFrame(
                {
                    "sentiment_polarity": np.zeros(len(texts), dtype=float),
                    "sentiment_intensity": np.zeros(len(texts), dtype=float),
                    "sentiment_uncertainty": np.ones(len(texts), dtype=float),
                }
            )

        tok, model = model_ref
        polarity_buf: list[np.ndarray] = []
        intensity_buf: list[np.ndarray] = []
        uncertainty_buf: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.cfg.batch_size):
                batch = texts[i : i + self.cfg.batch_size]
                enc = tok(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.cfg.max_length,
                    return_tensors="pt",
                )
                probs = torch.softmax(model(**enc).logits, dim=1).cpu().numpy()
                pol, inten, unc = self._to_dimensions_from_probs(probs)
                polarity_buf.append(pol)
                intensity_buf.append(inten)
                uncertainty_buf.append(unc)

        return pd.DataFrame(
            {
                "sentiment_polarity": np.concatenate(polarity_buf),
                "sentiment_intensity": np.concatenate(intensity_buf),
                "sentiment_uncertainty": np.concatenate(uncertainty_buf),
            }
        )

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        out = df.copy()
        texts = out[text_col].fillna("").astype(str).tolist() if text_col in out.columns else []
        dims = self.score_texts(texts)
        for col in dims.columns:
            out[col] = dims[col].values if len(dims) == len(out) else 0.0
        if "sentiment_uncertainty" not in out.columns:
            out["sentiment_uncertainty"] = 1.0
        return out


@dataclass
class PomPomRegimeConfig:
    n_components: int = 6
    random_state: int = 42
    positivity_quantile: float = 0.8
    volume_quantile: float = 0.8


class PomPomRegimeDetector:
    """
    GMM-based regime detector for IDX pump-and-dump conditions.
    POM_POM is assigned when a cluster has extreme positivity + anomalous volume.
    """

    def __init__(self, cfg: Optional[PomPomRegimeConfig] = None):
        self.cfg = cfg or PomPomRegimeConfig()
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(
            n_components=self.cfg.n_components,
            covariance_type="full",
            random_state=self.cfg.random_state,
            n_init=5,
        )
        self.feature_cols: list[str] = []
        self.pom_pom_clusters: set[int] = set()

    @staticmethod
    def _prepare_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["sentiment_polarity"] = pd.to_numeric(
            out.get("sentiment_polarity", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        out["sentiment_intensity"] = pd.to_numeric(
            out.get("sentiment_intensity", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        ).fillna(0.0)
        out["sentiment_uncertainty"] = pd.to_numeric(
            out.get("sentiment_uncertainty", pd.Series(index=out.index, dtype=float)),
            errors="coerce",
        ).fillna(1.0)
        out["volume_spike_z20"] = pd.to_numeric(
            out.get("volume_spike_z20", out.get("volume_z20", pd.Series(index=out.index, dtype=float))),
            errors="coerce",
        ).fillna(0.0)
        return out

    def fit_transform(self, df: pd.DataFrame, train_cutoff: Optional[pd.Timestamp] = None) -> tuple[pd.DataFrame, dict]:
        work = self._prepare_feature_columns(df)
        self.feature_cols = [
            "sentiment_polarity",
            "sentiment_intensity",
            "sentiment_uncertainty",
            "volume_spike_z20",
        ]
        X_all = work[self.feature_cols].astype(float).values

        if train_cutoff is not None and "date" in work.columns:
            date_mask = pd.to_datetime(work["date"], errors="coerce") <= pd.Timestamp(train_cutoff)
            train_idx = np.where(date_mask.fillna(False).values)[0]
            if len(train_idx) == 0:
                train_idx = np.arange(len(work))
        else:
            train_idx = np.arange(len(work))

        X_train = self.scaler.fit_transform(X_all[train_idx])
        self.gmm.fit(X_train)
        X_scaled = self.scaler.transform(X_all)
        probs = self.gmm.predict_proba(X_scaled)
        labels = self.gmm.predict(X_scaled)

        work["idx_regime_cluster_id"] = labels + 1
        work["idx_regime_cluster_conf"] = probs.max(axis=1)

        cluster_stats = (
            work.groupby("idx_regime_cluster_id", as_index=True)
            .agg(
                mean_polarity=("sentiment_polarity", "mean"),
                mean_volume_spike=("volume_spike_z20", "mean"),
            )
            .sort_index()
        )
        if len(cluster_stats) < 2:
            polarity_thr = float(work["sentiment_polarity"].quantile(self.cfg.positivity_quantile))
            volume_thr = float(work["volume_spike_z20"].quantile(self.cfg.volume_quantile))
        else:
            polarity_thr = float(cluster_stats["mean_polarity"].quantile(self.cfg.positivity_quantile))
            volume_thr = float(cluster_stats["mean_volume_spike"].quantile(self.cfg.volume_quantile))
        if np.isnan(polarity_thr):
            polarity_thr = 0.5
        if np.isnan(volume_thr):
            volume_thr = float(work["volume_spike_z20"].mean()) if len(work) else 0.0
        pom_clusters = cluster_stats[
            (cluster_stats["mean_polarity"] >= polarity_thr)
            & (cluster_stats["mean_volume_spike"] >= volume_thr)
        ].index.tolist()
        self.pom_pom_clusters = set(int(x) for x in pom_clusters)

        work["is_pom_pom_regime"] = work["idx_regime_cluster_id"].isin(self.pom_pom_clusters).astype(int)
        work["idx_regime_label"] = np.where(work["is_pom_pom_regime"] == 1, "POM_POM", "NORMAL")

        stats = {
            "idx_regime_features": self.feature_cols,
            "idx_regime_train_rows": int(len(train_idx)),
            "idx_regime_n_clusters": int(self.cfg.n_components),
            "idx_pom_pom_clusters": sorted(list(self.pom_pom_clusters)),
            "idx_polarity_threshold": polarity_thr,
            "idx_volume_threshold": volume_thr,
        }
        return work, stats
