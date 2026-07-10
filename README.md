# Algorithmic Bias in Hiring Systems

## Overview

This project demonstrates how machine learning systems can inherit bias from historical data.

The project compares three algorithms:

- Decision Tree
- Random Forest
- XGBoost

Although the algorithms differ, they all reproduce discrimination because the bias exists within the training data.

---

## Features

- Synthetic hiring dataset
- Classification using Decision Tree
- Random Forest model
- XGBoost model
- Fairness analysis
- Performance comparison
- Visualizations

---

## Dataset

Features include

- Gender
- Age
- Education
- Experience
- Interview Score
- Technical Score
- Communication Score

Labels

- Fair Selection
- Biased Selection

---

## Fairness Metrics

- Selection Rate
- Disparate Impact Ratio
- Statistical Parity Difference
- Equal Opportunity Difference
- False Positive Rate
- False Negative Rate

---

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run

Generate Dataset

```bash
python src/main_pipeline.py
```

Compare Algorithms

```bash
python src/03_algorithm_comparison.py
```

---

## Author

Tabish Khan

University of Kashmir

---

# Version 2 — Expanded Analysis

Everything above describes the original submission (v1). Based on
supervisor feedback, the project was expanded along every axis: dataset
size, protected attributes, algorithms, and mitigation strategies. v1's
files are untouched — v2 lives alongside them as additional scripts and
an additional dataset, so both versions remain runnable and comparable.

## What changed

| | v1 | v2 |
|---|---|---|
| Dataset size | 1,000 rows | **25,000 rows** |
| Protected attributes | Gender only | **Gender, Race/Ethnicity, Religion, Continent (Asia vs. rest)** |
| Algorithms | Decision Tree, Random Forest, XGBoost | **+ KNN, Logistic Regression, SVM, and a from-scratch NumPy Attention Network (6 classical + 1 attention = 7 total)** |
| Fairness analysis | Single-attribute (Gender) | **Single-attribute (all 4) + intersectional (Race x Gender, Religion x Continent)** |
| Mitigation | — | **Intersectional sample reweighting + group-specific threshold post-processing** |
| Literature grounding | — | **6 cited papers/standards, see REFERENCES.md** |

## New files (v2)

```
src/
├── 04_generate_dataset_v2.py     25,000-row dataset with 4 protected attributes
├── fairness_metrics_v2.py         generalized fairness module (any attribute + intersectional)
├── 05_algorithm_comparison_v2.py  6-model comparison across all 4 attributes
├── 06_attention_model.py          NumPy self-attention network, built from scratch
├── 07_intersectional_fairness.py  Race x Gender, Religion x Continent analysis
└── 08_mitigation_strategies.py    reweighting + group-specific thresholds

data/
├── hiring_dataset_v2.csv                        25,000-row expanded dataset
├── model_comparison_summary_v2.csv              all 6 models x all 4 attributes
├── intersectional_race_gender_v2.csv
├── intersectional_religion_continent_v2.csv
├── attention_feature_weights_v2.csv
└── mitigation_results_v2.csv

graphs/
├── fig17_perf_comparison_v2.png
├── fig18_fairness_heatmap_v2.png
├── fig19_selection_by_race_v2.png
├── fig20_selection_by_continent_v2.png
├── fig21_roc_overlay_v2.png
├── fig22_intersectional_heatmap.png
└── fig23_mitigation_comparison.png

REFERENCES.md   citations for every technique/metric borrowed from the literature
```

## Run v2

```bash
cd src
python3 04_generate_dataset_v2.py
python3 05_algorithm_comparison_v2.py
python3 06_attention_model.py
python3 07_intersectional_fairness.py
python3 08_mitigation_strategies.py
```

## Headline v2 findings

1. **All 7 models fail the EEOC four-fifths rule on every one of the 4
   protected attributes** — 28 out of 28 model x attribute combinations.
2. **Continent and Race/Ethnicity show the largest single-attribute bias**
   — larger than Gender for every model, e.g. XGBoost's Continent DP diff
   (0.204) exceeds its Gender DP diff (0.119).
3. **XGBoost is simultaneously the most accurate (96.5%) and the most
   biased (highest mean DP diff)** model — confirming and sharpening the
   v1 finding that more sophisticated algorithms encode historical
   discrimination more precisely, not less.
4. **Intersectional bias exceeds single-attribute bias**: the Race x
   Gender selection-rate gap (31.7 percentage points, Black+Female 0.1%
   vs. White+Male 31.8%) is larger than either the Race-only gap (21.1pp)
   or the Gender-only gap (13.3pp) — replicating the compounding-
   disadvantage pattern documented in Wilson & Caliskan (2024).
5. **The attention network's own attention weights name the problem
   directly**: Race_Ethnicity and Continent rank #1 and #2 by attention
   received, ahead of every legitimate merit feature (Technical_Score,
   Experience_Years, Interview_Score) — the model is quite literally
   paying more attention to protected attributes than to qualifications.
6. **Group-specific thresholds vastly outperform reweighting**: reweighting
   roughly halves demographic parity gaps (with a small accuracy cost);
   post-processing with per-group thresholds pushes every gap down to
   ~0.001-0.003, at the cost of needing group membership at
   decision-time — a real trade-off worth discussing in the report's
   ethics/limitations chapter (thresholding by protected attribute is
   itself legally contested in some jurisdictions, even when done to
   improve fairness).

See `REFERENCES.md` for the literature this expansion draws on.