from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SBERTEmbedder:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    normalize_embeddings: bool = True
    device: str | None = None

    def __post_init__(self) -> None:
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers could not be imported. Install the requirements in a fresh virtual environment."
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            raise RuntimeError(
                "SBERT model could not be loaded. Ensure the model name is correct and that the first run has internet access or a local cached model."
            ) from exc
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings)
