from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_COLUMNS = ["title", "text", "subject", "date"]


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)



def _standardize_frame(df: pd.DataFrame, binary_label: str) -> pd.DataFrame:
    working = df.copy()
    for column in RAW_COLUMNS:
        if column not in working.columns:
            working[column] = ""
        working[column] = working[column].fillna("").astype(str)
    working["binary_label"] = binary_label
    working["label_id"] = 1 if binary_label == "fake" else 0
    working["subject"] = (
        working["subject"]
        .str.replace(r"^[\"']+|[\"']+$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    working["raw_text"] = (
        working["title"].str.strip() + " " + working["text"].str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip()
    return working



def load_news_data(
    true_csv: str | Path,
    fake_csv: str | Path,
    *,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    true_df = pd.read_csv(true_csv,on_bad_lines='skip',engine='python')
    fake_df = pd.read_csv(fake_csv,on_bad_lines='skip',engine='python')

    true_std = _standardize_frame(true_df, "true")
    fake_std = _standardize_frame(fake_df, "fake")

    combined = pd.concat([true_std, fake_std], ignore_index=True)
    combined = combined[combined["raw_text"].str.len() > 0].reset_index(drop=True)
    combined["article_id"] = np.arange(len(combined))

    if drop_duplicates:
        combined = combined.drop_duplicates(subset=["title", "text"]).reset_index(drop=True)
        combined["article_id"] = np.arange(len(combined))

    return combined, true_std, fake_std



def build_dataset_audit(
    prepared_df: pd.DataFrame,
    true_df: pd.DataFrame,
    fake_df: pd.DataFrame,
) -> dict[str, Any]:
    raw_merged = pd.concat([true_df, fake_df], ignore_index=True)
    content_length = prepared_df["raw_text"].str.split().str.len()

    audit = {
        "prepared_rows": int(len(prepared_df)),
        "true_rows": int((prepared_df["binary_label"] == "true").sum()),
        "fake_rows": int((prepared_df["binary_label"] == "fake").sum()),
        "raw_rows_before_dedup": int(len(raw_merged)),
        "raw_true_rows": int(len(true_df)),
        "raw_fake_rows": int(len(fake_df)),
        "missing_titles_raw": int(raw_merged["title"].eq("").sum()),
        "duplicate_title_text_raw": int(raw_merged.duplicated(subset=["title", "text"]).sum()),
        "word_count_summary": {
            "mean": float(content_length.mean()),
            "median": float(content_length.median()),
            "min": int(content_length.min()),
            "max": int(content_length.max()),
        },
        "top_subjects_true": true_df["subject"].value_counts().head(10).to_dict(),
        "top_subjects_fake": fake_df["subject"].value_counts().head(10).to_dict(),
    }
    return audit



def make_balanced_sample(
    df: pd.DataFrame,
    *,
    per_class: int,
    random_state: int = 42,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for label, group in df.groupby("binary_label"):
        take = min(per_class, len(group))
        chunks.append(group.sample(n=take, random_state=random_state))
    sampled = pd.concat(chunks, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    sampled["article_id"] = np.arange(len(sampled))
    return sampled
