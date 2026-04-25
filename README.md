# Fake News Classification Project

This project builds a complete fake-news intelligence pipeline with four layers:

1. **spaCy cleaning** for normalization, token filtering, and lemmatized text.
2. **SBERT embeddings** for dense semantic representation of each article.
3. **Clustering on fake-news articles** to generate two extra labels:
   - `fake_type_label` from semantic article clusters
   - `severity_label` from clustered sensationalism and distortion features
4. **Final supervised classifiers** using the generated labels:
   - Logistic Regression
   - Linear SVM
   - XGBoost

The project is designed for your uploaded `True.csv` and `Fake_Parsable.csv`, but it also works on the included demo sample.

## Project idea

The pipeline is intentionally **two-stage**:

- Stage A discovers structure inside fake news with **unsupervised clustering**.
- Stage B trains fast supervised models so inference becomes cheap and repeatable.

That gives you a richer output than plain binary classification:

- **Is the article fake or true?**
- **If fake, what type of fake news does it look like?**
- **How severe is the fake article?**

## Folder structure

```text
fake_news_sbert_clustering_project/
├── README.md
├── requirements.txt
├── run_pipeline.py
├── src/
│   ├── data_utils.py
│   ├── preprocess.py
│   ├── embeddings.py
│   ├── cluster_labels.py
│   ├── train_models.py
│   └── inference.py
├── docs/
│   └── project_report.md
├── data/
│   └── sample/
│       ├── True_demo.csv
│       ├── Fake_demo.csv
│       └── merged_demo_sample.csv
└── artifacts/
    └── dataset_audit.json
```

## How severity and fake-type labels are created

### 1) Fake type label

Only the **fake-news subset** is used.

- Cleaned fake articles are embedded with SBERT.
- `KMeans` searches for the best `k` using cosine silhouette score.
- Each cluster gets a human-readable name from its top TF-IDF keywords.
- Example style of generated labels:
  - `type_0_trump_white_house`
  - `type_1_clinton_email_fbi`
  - `type_2_syria_russia_war`

### 2) Severity label

Again, only fake articles are used.

A feature bank is built from the raw title and body:

- sensational-word ratio
- title sensational-word ratio
- exclamation ratio
- question ratio
- ALL-CAPS ratio
- quote ratio
- elongated-token ratio
- digit ratio
- normalized title length
- normalized body length

Those features are clustered into **3 groups** and then ranked by a composite sensationalism proxy to map the clusters to:

- `low`
- `medium`
- `high`

## Installation

Create a clean environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended one-time model setup:

```bash
python -m spacy download en_core_web_sm
```

> If `en_core_web_sm` is not installed, the project falls back to `spacy.blank("en")`.
> For the first SBERT run, internet is normally needed once so the embedding model can be downloaded and cached.

## Running on your full dataset

Put the CSV files somewhere accessible, then run:

```bash
python run_pipeline.py \
  --true_csv /path/to/True.csv \
  --fake_csv /path/to/Fake_Parsable.csv \
  --output_dir artifacts/generated \
  --sbert_model all-MiniLM-L6-v2 \
  --min_fake_type_k 4 \
  --max_fake_type_k 8
```

## Quick demo run

```bash
python run_pipeline.py \
  --true_csv data/sample/True_demo.csv \
  --fake_csv data/sample/Fake_demo.csv \
  --sample_size_per_class 50 \
  --min_fake_type_k 2 \
  --max_fake_type_k 3
```

The quick demo is only for code sanity checks. For real experiments, use the actual `True.csv` and `Fake_Parsable.csv` files.

## Main outputs

After a run, the output directory contains:

- `dataset_audit.json`
- `cleaned_dataset.csv`
- `sbert_embeddings.npy`
- `fake_type_cluster_summary.json`
- `severity_cluster_summary.json`
- `dataset_with_generated_labels.csv`
- `binary/leaderboard.csv`
- `fake_type/leaderboard.csv`
- `severity/leaderboard.csv`
- confusion matrices for the best model in each task
- PCA cluster plots

## Inference after training

```python
from src.inference import FakeNewsPredictor

predictor = FakeNewsPredictor("artifacts/generated")
result = predictor.predict(
    title="Breaking: secret report shocks Washington",
    text="Long article body here..."
)
print(result)
```

The inference output contains:

- `binary_label`
- `fake_type_label`
- `severity_label`
- `clean_text`

## Suggested viva explanation

A clean way to explain this project in an academic setting:

1. *spaCy* reduces lexical noise.
2. *SBERT* converts each article into a semantic vector.
3. *Clustering* discovers hidden fake-news themes and severity bands.
4. *Classifiers* learn to reproduce those labels quickly at inference time.
5. The result is a **hierarchical fake-news detector**, not just a yes/no classifier.

## Possible improvements

- Replace the keyword naming step with BERTopic or c-TF-IDF summaries.
- Add explainability with SHAP for XGBoost.
- Add calibration and probability thresholds.
- Add a small Streamlit demo.
- Compare SBERT with DeBERTa or domain-specific transformer embeddings.
