from __future__ import annotations

import html
import re
from typing import Iterable, Sequence

import spacy


KEEP_NEGATIONS = {"no", "not", "never"}


class SpacyCleaner:
    """Lightweight spaCy based cleaner with a graceful fallback path.

    The preferred path uses ``en_core_web_sm``. If the model is not installed,
    the cleaner falls back to ``spacy.blank('en')`` with a lookup lemmatizer
    when available.
    """

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        disable: Sequence[str] | None = None,
    ) -> None:
        disable = tuple(disable or ["parser", "ner", "textcat"])
        self.nlp, self.backend_name = self._load_pipeline(model_name, disable)
        self.nlp.max_length = max(self.nlp.max_length, 5_000_000)

    @staticmethod
    def normalize_raw_text(text: str) -> str:
        text = html.unescape(str(text or ""))
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[\r\n\t]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load_pipeline(self, model_name: str, disable: Sequence[str]):
        try:
            nlp = spacy.load(model_name, disable=list(disable))
            if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp, model_name
        except Exception:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            try:
                if "lemmatizer" not in nlp.pipe_names:
                    nlp.add_pipe("lemmatizer", config={"mode": "lookup"})
                nlp.initialize()
                backend = "spacy.blank('en') + lookup lemmatizer"
            except Exception:
                backend = "spacy.blank('en')"
            return nlp, backend

    def clean_document(self, text: str) -> str:
        doc = self.nlp(self.normalize_raw_text(text))
        cleaned_tokens: list[str] = []
        for token in doc:
            if token.is_space or token.is_punct or token.like_url or token.like_email:
                continue
            if token.is_stop and token.lower_ not in KEEP_NEGATIONS:
                continue
            candidate = token.lemma_.lower().strip() if getattr(token, "lemma_", "") else token.text.lower().strip()
            if candidate in {"", "-pron-"}:
                candidate = token.text.lower().strip()
            candidate = re.sub(r"[^a-zA-Z']+", "", candidate)
            if not candidate or candidate.isdigit() or len(candidate) <= 2:
                continue
            cleaned_tokens.append(candidate)
        return " ".join(cleaned_tokens)

    def clean_corpus(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 128,
        n_process: int = 1,
    ) -> list[str]:
        normalized = [self.normalize_raw_text(text) for text in texts]
        docs = self.nlp.pipe(normalized, batch_size=batch_size, n_process=n_process)
        return [self.clean_document_from_doc(doc) for doc in docs]

    def clean_document_from_doc(self, doc) -> str:
        cleaned_tokens: list[str] = []
        for token in doc:
            if token.is_space or token.is_punct or token.like_url or token.like_email:
                continue
            if token.is_stop and token.lower_ not in KEEP_NEGATIONS:
                continue
            candidate = token.lemma_.lower().strip() if getattr(token, "lemma_", "") else token.text.lower().strip()
            if candidate in {"", "-pron-"}:
                candidate = token.text.lower().strip()
            candidate = re.sub(r"[^a-zA-Z']+", "", candidate)
            if not candidate or candidate.isdigit() or len(candidate) <= 2:
                continue
            cleaned_tokens.append(candidate)
        return " ".join(cleaned_tokens)
