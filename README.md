


# Algorithmic Bias in Everyday Systems

**Group D: Nexus Thinkers** | University of Kashmir Institute of Technology, Zakura Campus
FYUGP Data Science, Semester III | Mentor: Dr. Nahida Shafi | Co-supervisor: Dr. Mueed Hafiz

## Overview

This project demonstrates how machine learning systems can inherit and
amplify bias from historical data, using a synthetic hiring dataset as a
case study. It progresses in two phases:

- **v1** — a single-attribute (Gender) demonstration on 1,000 rows,
  comparing three classifiers.
- **v2** — an expanded, multi-attribute study on 25,000 rows, adding
  Race/Ethnicity, Religion, and Continent as protected attributes,
  intersectional fairness analysis, a from-scratch custom classifier
  (FACC), and two bias-mitigation strategies.

---

## Features

**v1 (single-attribute baseline)**
- Synthetic hiring dataset (1,000 rows, Gender only)
- Decision Tree classifier
- Fairness metrics (selection rate, disparate impact, statistical parity)

**v2 (expanded study)**
- Expanded synthetic dataset (25,000 rows) with four protected attributes:
  Gender, Race/Ethnicity, Religion, Continent
- Six algorithms compared: Decision Tree, Random Forest, XGBoost,
  K-Nearest Neighbours, Logistic Regression, Support Vector Machine
- **FACC (Fairness-Aware Custom Classifier)** — a feature-wise
  classifier built from scratch in NumPy, using a scaled dot-product
  weighting mechanism (in the spirit of Transformer self-attention,
  Vaswani et al. 2017) to produce interpretable per-feature relevance
  weights — see dissertation Section 6.11 for the full theoretical
  grounding
- Intersectional fairness analysis (e.g. Race x Gender, Religion x
  Continent) — checking for compounded disadvantage that single-attribute
  audits miss
- Bias mitigation: sample reweighting (Kamiran & Calders, 2012) and
  group-specific decision thresholds
- Fairness metrics: Selection Rate, Disparate Impact Ratio, Statistical
  Parity Difference, Equal Opportunity Difference, False Positive/Negative
  Rate, Four-Fifths Rule check
- Visualizations: performance comparison, fairness heatmaps, ROC overlays,
  intersectional heatmaps, mitigation comparison (11 figures total)

---

## Dataset

**Features**
- Gender, Age, Education, Experience, Interview Score, Technical Score,
  Communication Score
- *(v2 adds)* Race/Ethnicity, Religion, Continent

**Labels**
- `Selected_Fair` — selection based only on merit
- `Selected` — selection after the injected bias penalty (what the models
  are trained on)

---

## Fairness Metrics

- Selection Rate
- Disparate Impact Ratio (Four-Fifths Rule)
- Statistical Parity Difference
- Equal Opportunity Difference
- False Positive Rate / False Negative Rate

---

## Technologies

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn

---

## Installation

```
pip install -r requirements.txt
```

---

## Project Structure

```
src/
  main_pipeline.py              v1: dataset generation + Decision Tree + fairness metrics (all-in-one)
  03_algorithm_comparison.py    v1: Decision Tree / Random Forest / XGBoost comparison, Gender only
  04_generate_dataset_v2.py     v2: generates the 25,000-row, 4-attribute dataset
  05_algorithm_comparison_v2.py v2: 6-algorithm comparison across 4 protected attributes
  06_facc_model.py              v2: FACC -- custom NumPy classifier with per-feature relevance weights
  07_intersectional_fairness.py v2: intersectional (multi-attribute) fairness analysis
  08_mitigation_strategies.py   v2: reweighting + group-specific threshold mitigation on XGBoost
  fairness_metrics.py           v1 fairness metric helpers
  fairness_metrics_v2.py        v2 fairness metric helpers (multi-attribute)
data/                           generated datasets and results (CSV)
graphs/                         generated figures (PNG)
```

## Run

**v1 (baseline, single script):**
```
python src/main_pipeline.py
```

**v2 (full pipeline, run in order):**
```
python src/04_generate_dataset_v2.py
python src/05_algorithm_comparison_v2.py
python src/06_facc_model.py
python src/07_intersectional_fairness.py
python src/08_mitigation_strategies.py
```

Each v2 script after `04` depends on the dataset (and, for `07`/`08`, the
outputs of `05`) having been generated first -- run them in the order
above.

---

## Author

Tabish Khan
University of Kashmir Institute of Technology
