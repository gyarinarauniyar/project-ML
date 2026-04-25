from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from src.cluster_labels import (
    build_fake_type_labels,
    build_severity_labels,
    save_cluster_artifacts,
    save_cluster_projection,
)
from src.data_utils import (
    build_dataset_audit,
    ensure_dir,
    load_news_data,
    make_balanced_sample,
    save_json,
)
from src.embeddings import SBERTEmbedder
from src.preprocess import SpacyCleaner
from src.train_models import train_task



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End to end fake news project with spaCy cleaning, SBERT embeddings, clustering labels, and classical classifiers.",
    )
    parser.add_argument("--true_csv", required=True, help="Path to the true-news CSV file")
    parser.add_argument("--fake_csv", required=True, help="Path to the fake-news CSV file")
    parser.add_argument("--output_dir", default="artifacts/generated", help="Directory where all outputs are saved")
    parser.add_argument("--spacy_model", default="en_core_web_sm", help="spaCy pipeline name")
    parser.add_argument("--sbert_model", default="all-MiniLM-L6-v2", help="SBERT model name or local path")
    parser.add_argument("--sample_size_per_class", type=int, default=None, help="Optional balanced sample size per class")
    parser.add_argument("--clean_batch_size", type=int, default=128, help="spaCy pipe batch size")
    parser.add_argument("--embed_batch_size", type=int, default=64, help="SBERT encode batch size")
    parser.add_argument("--min_fake_type_k", type=int, default=4, help="Minimum number of fake-type clusters")
    parser.add_argument("--max_fake_type_k", type=int, default=8, help="Maximum number of fake-type clusters")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    df, true_df, fake_df = load_news_data(args.true_csv, args.fake_csv, drop_duplicates=True)
    audit = build_dataset_audit(df, true_df, fake_df)
    save_json(audit, output_dir / "dataset_audit.json")

    if args.sample_size_per_class:
        df = make_balanced_sample(df, per_class=args.sample_size_per_class, random_state=args.random_state)

    cleaner = SpacyCleaner(args.spacy_model)
    print(f"[1/6] Cleaning text with spaCy backend: {cleaner.backend_name}")
    df["clean_text"] = cleaner.clean_corpus(
        df["raw_text"].tolist(),
        batch_size=args.clean_batch_size,
        n_process=1,
    )
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    df.to_csv(output_dir / "cleaned_dataset.csv", index=False)

    print(f"[2/6] Encoding text with SBERT model: {args.sbert_model}")
    embedder = SBERTEmbedder(
        model_name=args.sbert_model,
        batch_size=args.embed_batch_size,
        normalize_embeddings=True,
    )
    embeddings = embedder.encode(df["clean_text"].tolist())
    np.save(output_dir / "sbert_embeddings.npy", embeddings)

    fake_mask = df["binary_label"].eq("fake")
    fake_embeddings = embeddings[fake_mask.to_numpy()]
    fake_texts = df.loc[fake_mask, "clean_text"].tolist()
    fake_titles = df.loc[fake_mask, "title"].tolist()
    fake_raw_texts = df.loc[fake_mask, "raw_text"].tolist()

    print("[3/6] Building fake-type clusters")
    fake_type = build_fake_type_labels(
        fake_embeddings,
        fake_texts,
        min_k=args.min_fake_type_k,
        max_k=args.max_fake_type_k,
        random_state=args.random_state,
    )
    df.loc[fake_mask, "fake_type_label"] = fake_type["cluster_name_labels"]
    df.loc[~fake_mask, "fake_type_label"] = "not_fake"
    save_json(
        {
            "best_k": fake_type["best_k"],
            "silhouette_scores": fake_type["silhouette_scores"],
            "cluster_keywords": fake_type["cluster_keywords"],
            "cluster_name_map": fake_type["cluster_name_map"],
        },
        output_dir / "fake_type_cluster_summary.json",
    )
    save_cluster_artifacts(fake_type["clusterer"], output_dir / "fake_type_clusterer.joblib")
    save_cluster_projection(
        fake_embeddings,
        fake_type["cluster_ids"],
        output_dir / "fake_type_clusters_pca.png",
        title="Fake-type clusters (PCA projection)",
        random_state=args.random_state,
    )

    print("[4/6] Building severity labels")
    severity = build_severity_labels(fake_titles, fake_raw_texts, random_state=args.random_state)
    df.loc[fake_mask, "severity_label"] = severity["severity_labels"]
    df.loc[~fake_mask, "severity_label"] = "not_fake"
    severity["feature_frame"].to_csv(output_dir / "severity_features.csv", index=False)
    severity["cluster_feature_means"].to_csv(output_dir / "severity_cluster_feature_means.csv", index=False)
    save_json(
        {"severity_name_map": severity["severity_name_map"]},
        output_dir / "severity_cluster_summary.json",
    )
    save_cluster_artifacts(severity["clusterer"], output_dir / "severity_clusterer.joblib")
    save_cluster_artifacts(severity["scaler"], output_dir / "severity_scaler.joblib")

    print("[5/6] Training classifiers")
    binary_summary = train_task(
        task_name="binary",
        X=embeddings,
        y=df["binary_label"],
        output_dir=output_dir / "binary",
        test_size=args.test_size,
        random_state=args.random_state,
    )
    fake_type_summary = train_task(
        task_name="fake_type",
        X=fake_embeddings,
        y=df.loc[fake_mask, "fake_type_label"],
        output_dir=output_dir / "fake_type",
        test_size=args.test_size,
        random_state=args.random_state,
    )
    severity_summary = train_task(
        task_name="severity",
        X=fake_embeddings,
        y=df.loc[fake_mask, "severity_label"],
        output_dir=output_dir / "severity",
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("[6/6] Saving merged dataset with generated labels")
    ordered_columns = [
        "article_id",
        "binary_label",
        "title",
        "text",
        "subject",
        "date",
        "raw_text",
        "clean_text",
        "fake_type_label",
        "severity_label",
    ]
    df[ordered_columns].to_csv(output_dir / "dataset_with_generated_labels.csv", index=False)

    save_json(
        {
            "binary": binary_summary,
            "fake_type": fake_type_summary,
            "severity": severity_summary,
            "sbert_model": args.sbert_model,
            "spacy_backend": cleaner.backend_name,
            "rows_used": int(len(df)),
        },
        output_dir / "run_summary.json",
    )

    print("Pipeline finished successfully.")
    print(f"Artifacts saved in: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
