from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier



def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def _write_json(payload: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)



def build_model_zoo(num_classes: int, random_state: int = 42) -> dict[str, Any]:
    objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
    eval_metric = "logloss" if num_classes == 2 else "mlogloss"
    xgb_kwargs = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "tree_method": "hist",
        "objective": objective,
        "eval_metric": eval_metric,
        "random_state": random_state,
        "n_jobs": 1,
        "verbosity": 0,
    }
    if num_classes > 2:
        xgb_kwargs["num_class"] = num_classes

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            random_state=random_state,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            dual=False,
            max_iter=4000,
            random_state=random_state,
        ),
        "xgboost": XGBClassifier(**xgb_kwargs),
    }



def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    path: str | Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
        colorbar=False,
    )
    disp.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()



def train_task(
    *,
    task_name: str,
    X: np.ndarray,
    y: list[str] | np.ndarray | pd.Series,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    output_dir = _ensure_dir(output_dir)
    y_series = pd.Series(y).astype(str)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_series)
    class_names = label_encoder.classes_.tolist()

    if len(class_names) < 2:
        raise ValueError(f"Task '{task_name}' needs at least two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded,
    )

    np.save(output_dir / "X_test.npy", X_test)
    np.save(output_dir / "y_test.npy", y_test)
    _write_json(
        {
            "task_name": task_name,
            "class_names": class_names,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
        },
        output_dir / "split_info.json",
    )

    results: list[dict[str, Any]] = []
    best_model_name = ""
    best_macro_f1 = -1.0
    best_bundle: dict[str, Any] | None = None
    best_predictions: np.ndarray | None = None

    for model_name, model in build_model_zoo(len(class_names), random_state=random_state).items():
        if model_name == "xgboost":
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = float(accuracy_score(y_test, predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
        result = {
            "task_name": task_name,
            "model_name": model_name,
            "accuracy": accuracy,
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        }
        results.append(result)

        report = classification_report(
            y_test,
            predictions,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        )
        _write_json(report, output_dir / f"{model_name}_classification_report.json")
        joblib.dump(
            {
                "model": model,
                "label_encoder": label_encoder,
                "task_name": task_name,
                "class_names": class_names,
            },
            output_dir / f"{model_name}_bundle.joblib",
        )

        if result["macro_f1"] > best_macro_f1:
            best_macro_f1 = result["macro_f1"]
            best_model_name = model_name
            best_bundle = {
                "model": model,
                "label_encoder": label_encoder,
                "task_name": task_name,
                "class_names": class_names,
            }
            best_predictions = predictions

    leaderboard = pd.DataFrame(results).sort_values(
        by=["macro_f1", "accuracy"], ascending=False
    )
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)

    if best_bundle is None or best_predictions is None:
        raise RuntimeError(f"No model was trained for task '{task_name}'.")

    joblib.dump(best_bundle, output_dir / "best_model_bundle.joblib")
    _save_confusion_matrix(
        y_test,
        best_predictions,
        class_names,
        output_dir / "best_model_confusion_matrix.png",
        title=f"{task_name} | best model = {best_model_name}",
    )

    summary = {
        "task_name": task_name,
        "best_model": best_model_name,
        "best_macro_f1": float(best_macro_f1),
        "class_names": class_names,
        "leaderboard_path": str(output_dir / "leaderboard.csv"),
    }
    _write_json(summary, output_dir / "summary.json")
    return summary
