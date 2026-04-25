from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


SENSATIONAL_LEXICON = {
    "alert", "amazing", "betrayal", "bombshell", "chaos", "conspiracy",
    "coverup", "crisis", "destroy", "disaster", "explosive", "exposed",
    "fake", "fraud", "hoax", "leaked", "lies", "outrage", "panic",
    "scandal", "secret", "shocking", "traitor", "unbelievable", "urgent",
    "warning", "wild", "corrupt", "rigged", "agenda", "propaganda",
}



def _safe_mean(vectorized_matrix) -> np.ndarray:
    return np.asarray(vectorized_matrix.mean(axis=0)).ravel()



def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return value or "cluster"



def select_best_k(
    embeddings: np.ndarray,
    *,
    min_k: int = 4,
    max_k: int = 8,
    random_state: int = 42,
    sample_size: int = 5000,
) -> tuple[int, list[dict[str, Any]]]:
    if min_k < 2:
        raise ValueError("min_k must be at least 2")
    if max_k < min_k:
        raise ValueError("max_k must be >= min_k")

    if len(embeddings) <= sample_size:
        probe = embeddings
    else:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(embeddings), size=sample_size, replace=False)
        probe = embeddings[indices]

    scores: list[dict[str, Any]] = []
    best_score = -math.inf
    best_k = min_k
    for k in range(min_k, max_k + 1):
        clusterer = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )
        labels = clusterer.fit_predict(probe)
        if len(np.unique(labels)) < 2:
            score = -1.0
        else:
            score = float(silhouette_score(probe, labels, metric="cosine"))
        scores.append({"k": k, "silhouette_cosine": score})
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, scores



def extract_cluster_keywords(
    texts: list[str],
    labels: np.ndarray,
    *,
    top_n: int = 6,
) -> dict[int, list[str]]:
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3 if len(texts) > 500 else 1,
        max_df=0.85,
        max_features=20000,
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    keyword_map: dict[int, list[str]] = {}
    for cluster_id in sorted(np.unique(labels)):
        cluster_matrix = matrix[labels == cluster_id]
        mean_scores = _safe_mean(cluster_matrix)
        top_indices = mean_scores.argsort()[::-1][:top_n]
        keyword_map[int(cluster_id)] = feature_names[top_indices].tolist()
    return keyword_map



def build_fake_type_labels(
    fake_embeddings: np.ndarray,
    fake_texts: list[str],
    *,
    min_k: int = 4,
    max_k: int = 8,
    random_state: int = 42,
) -> dict[str, Any]:
    best_k, silhouette_scores = select_best_k(
        fake_embeddings,
        min_k=min_k,
        max_k=max_k,
        random_state=random_state,
    )
    clusterer = KMeans(
        n_clusters=best_k,
        random_state=random_state,
        n_init=10,
    )
    cluster_ids = clusterer.fit_predict(fake_embeddings)
    keyword_map = extract_cluster_keywords(fake_texts, cluster_ids)
    name_map = {
        cluster_id: f"type_{cluster_id}_{slugify('_'.join(keywords[:3]))}"
        for cluster_id, keywords in keyword_map.items()
    }
    name_labels = [name_map[int(idx)] for idx in cluster_ids]
    return {
        "best_k": best_k,
        "silhouette_scores": silhouette_scores,
        "cluster_ids": cluster_ids,
        "cluster_keywords": keyword_map,
        "cluster_name_map": name_map,
        "cluster_name_labels": name_labels,
        "clusterer": clusterer,
    }



def _tokenize_alpha(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", str(text or ""))



def compute_severity_feature_frame(
    titles: list[str],
    raw_texts: list[str],
) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for title, raw_text in zip(titles, raw_texts, strict=True):
        text = str(raw_text or "")
        title = str(title or "")
        words = _tokenize_alpha(text.lower())
        title_words = _tokenize_alpha(title.lower())
        word_count = max(len(words), 1)
        title_word_count = max(len(title_words), 1)
        all_caps_tokens = re.findall(r"\b[A-Z]{2,}\b", text)
        elongated_tokens = re.findall(r"\b\w*(\w)\1{2,}\w*\b", text.lower())
        sensational_hits = sum(token in SENSATIONAL_LEXICON for token in words)
        title_sensational_hits = sum(token in SENSATIONAL_LEXICON for token in title_words)
        digit_hits = len(re.findall(r"\b\d+\b", text))
        quote_count = text.count('"') + text.count("'")
        record = {
            "sensational_ratio": sensational_hits / word_count,
            "title_sensational_ratio": title_sensational_hits / title_word_count,
            "exclamation_ratio": text.count("!") / word_count,
            "question_ratio": text.count("?") / word_count,
            "all_caps_ratio": len(all_caps_tokens) / word_count,
            "quote_ratio": quote_count / max(len(text), 1),
            "elongated_ratio": len(elongated_tokens) / word_count,
            "digit_ratio": digit_hits / word_count,
            "title_length_norm": title_word_count / 20.0,
            "body_length_norm": min(word_count, 2000) / 1000.0,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)



def build_severity_labels(
    titles: list[str],
    raw_texts: list[str],
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    features = compute_severity_feature_frame(titles, raw_texts)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    clusterer = KMeans(n_clusters=3, random_state=random_state, n_init=20)
    cluster_ids = clusterer.fit_predict(scaled)

    feature_means = features.groupby(cluster_ids).mean(numeric_only=True)
    severity_proxy = (
        0.40 * feature_means["sensational_ratio"]
        + 0.15 * feature_means["title_sensational_ratio"]
        + 0.15 * feature_means["exclamation_ratio"]
        + 0.10 * feature_means["question_ratio"]
        + 0.10 * feature_means["all_caps_ratio"]
        + 0.05 * feature_means["quote_ratio"]
        + 0.05 * feature_means["elongated_ratio"]
    )
    ranked_clusters = severity_proxy.sort_values().index.tolist()
    name_map = {
        int(ranked_clusters[0]): "low",
        int(ranked_clusters[1]): "medium",
        int(ranked_clusters[2]): "high",
    }
    labels = [name_map[int(cluster_id)] for cluster_id in cluster_ids]
    return {
        "feature_frame": features,
        "cluster_ids": cluster_ids,
        "severity_labels": labels,
        "severity_name_map": name_map,
        "clusterer": clusterer,
        "scaler": scaler,
        "cluster_feature_means": feature_means.reset_index().rename(columns={"index": "cluster_id"}),
    }



def save_cluster_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: str | Path,
    title: str,
    *,
    random_state: int = 42,
    max_points: int = 3000,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(embeddings) > max_points:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(embeddings), size=max_points, replace=False)
        plot_embeddings = embeddings[indices]
        plot_labels = labels[indices]
    else:
        plot_embeddings = embeddings
        plot_labels = labels

    reducer = PCA(n_components=2, random_state=random_state)
    coords = reducer.fit_transform(plot_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=plot_labels, alpha=0.65, s=18)
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()



def save_cluster_artifacts(clusterer: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clusterer, path)
