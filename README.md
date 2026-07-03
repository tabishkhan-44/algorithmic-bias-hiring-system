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