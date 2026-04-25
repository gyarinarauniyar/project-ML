from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from .embeddings import SBERTEmbedder
from .preprocess import SpacyCleaner


@dataclass
class FakeNewsPredictor:
    artifact_dir: str | Path
    sbert_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        self.cleaner = SpacyCleaner(self.spacy_model)
        self.embedder = SBERTEmbedder(self.sbert_model)
        self.binary_bundle = joblib.load(self.artifact_dir / "binary" / "best_model_bundle.joblib")
        self.fake_type_bundle = joblib.load(self.artifact_dir / "fake_type" / "best_model_bundle.joblib")
        self.severity_bundle = joblib.load(self.artifact_dir / "severity" / "best_model_bundle.joblib")

    def _predict_from_bundle(self, bundle: dict[str, Any], embedding):
        encoded_prediction = bundle["model"].predict(embedding)
        return bundle["label_encoder"].inverse_transform(encoded_prediction)[0]

    def predict(self, *, title: str, text: str) -> dict[str, Any]:
        raw_text = f"{title} {text}".strip()
        clean_text = self.cleaner.clean_document(raw_text)
        embedding = self.embedder.encode([clean_text])

        binary_label = self._predict_from_bundle(self.binary_bundle, embedding)
        result = {
            "binary_label": binary_label,
            "clean_text": clean_text,
            "fake_type_label": "not_fake",
            "severity_label": "not_fake",
        }
        if binary_label == "fake":
            result["fake_type_label"] = self._predict_from_bundle(self.fake_type_bundle, embedding)
            result["severity_label"] = self._predict_from_bundle(self.severity_bundle, embedding)
        return result
