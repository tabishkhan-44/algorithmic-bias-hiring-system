"""
=============================================================================
Algorithmic Bias in Everyday Systems — Dataset v2 (Expanded)
=============================================================================
Group D: Nexus Thinkers | University of Kashmir IT | Semester III, 2025

WHY V2?
v1 (main_pipeline.py) demonstrated bias along a SINGLE protected attribute
(Gender) on 1,000 rows. Supervisor feedback asked for a larger, richer
dataset that also captures race/ethnicity, religion, and nationality
(continent) bias — the three attributes most studied in the algorithmic
hiring fairness literature (Fabris et al., "Fairness and Bias in
Algorithmic Hiring: A Multidisciplinary Survey", ACM TIST 2024).

This script keeps v1's core design (a Fair_Score built only from merit
features, and a Biased_Score = Fair_Score reduced by a group-specific
multiplicative penalty) but:
  1. Scales N from 1,000 to 25,000.
  2. Adds Race_Ethnicity, Religion, and Continent as independent protected
     attributes, each with its own documented penalty.
  3. Combines all penalties MULTIPLICATIVELY per candidate, so a candidate
     disadvantaged on several attributes at once (e.g. a Black, Muslim,
     Asia-based, Non-Binary candidate) receives a compounded penalty. This
     directly models "intersectional" disadvantage described in Wilson &
     Caliskan (2024) and the Fabris et al. (2024) survey — bias doesn't
     just add up per attribute, it compounds for people at the
     intersection of several disadvantaged groups.

Run:  python3 04_generate_dataset_v2.py
Output: ../data/hiring_dataset_v2.csv  (25,000 rows)
=============================================================================
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 25000

# ── STEP 1: DEMOGRAPHICS ────────────────────────────────────────────────
gender = np.random.choice(['Male', 'Female', 'Non-Binary'], N, p=[0.49, 0.47, 0.04])

race_ethnicity = np.random.choice(
    ['White', 'Black', 'Hispanic', 'Asian', 'Other'], N,
    p=[0.55, 0.14, 0.14, 0.12, 0.05])

religion = np.random.choice(
    ['Christian', 'Muslim', 'Hindu', 'Jewish', 'Buddhist', 'Unaffiliated'], N,
    p=[0.45, 0.16, 0.12, 0.05, 0.06, 0.16])

continent = np.random.choice(
    ['Asia', 'North America', 'Europe', 'Africa', 'South America', 'Oceania'], N,
    p=[0.34, 0.24, 0.20, 0.10, 0.08, 0.04])

age = np.random.normal(33, 10, N).clip(21, 65).astype(int)

education = np.random.choice(
    ['High School', 'Bachelor', 'Master', 'PhD'], N,
    p=[0.15, 0.50, 0.28, 0.07])

experience = np.random.gamma(3.0, 1.6, N).clip(0, 40).round(1)
interview_score = np.random.normal(70, 13, N).clip(0, 100).round(1)
technical_score = np.random.normal(70, 14, N).clip(0, 100).round(1)
communication_score = np.random.normal(69, 12, N).clip(0, 100).round(1)

edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
edu_num = np.array([edu_map[e] for e in education])


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


# ── STEP 2: FAIR SCORE (merit only — no protected attributes involved) ──
fair_score = (
    0.25 * normalize(edu_num) +
    0.20 * normalize(experience) +
    0.20 * normalize(interview_score) +
    0.25 * normalize(technical_score) +
    0.10 * normalize(communication_score)
)

# ── STEP 3: GROUP-SPECIFIC BIAS PENALTIES (documented, citable) ─────────
# Each dict below is the "ground truth" discrimination baked into the
# data, expressed as a multiplier applied to the fair score (1.0 = no
# penalty). Magnitudes are illustrative, chosen to be large enough to be
# clearly measurable by every model, in line with how prior work
# (Kamiran & Calders, 2012; Bogen & Rieke, 2018) simulates discriminatory
# training labels for fairness research.
GENDER_PENALTY = {'Male': 1.00, 'Female': 0.88, 'Non-Binary': 0.93}
RACE_PENALTY = {'White': 1.00, 'Asian': 0.90, 'Other': 0.85, 'Hispanic': 0.80, 'Black': 0.75}
RELIGION_PENALTY = {'Christian': 1.00, 'Unaffiliated': 0.97, 'Buddhist': 0.95,
                     'Jewish': 0.95, 'Hindu': 0.92, 'Muslim': 0.80}
CONTINENT_PENALTY = {'North America': 1.00, 'Europe': 1.00, 'Oceania': 1.00,
                      'South America': 0.88, 'Asia': 0.85, 'Africa': 0.80}


def age_penalty(a):
    # Reversed-direction age bias, replicated from the v1 findings: once
    # merit is controlled for, the sharpest penalty falls on 60+, not the
    # very young — a deliberately "surprising" result for the discussion
    # chapter (see README, "Age confound / Simpson's paradox" finding).
    if a < 25:
        return 0.97
    elif a <= 44:
        return 1.02
    elif a <= 59:
        return 0.90
    else:
        return 0.80


gender_pen = np.array([GENDER_PENALTY[g] for g in gender])
race_pen = np.array([RACE_PENALTY[r] for r in race_ethnicity])
religion_pen = np.array([RELIGION_PENALTY[r] for r in religion])
continent_pen = np.array([CONTINENT_PENALTY[c] for c in continent])
age_pen = np.array([age_penalty(a) for a in age])

# Intersectional / compounding penalty: multiply every attribute's
# penalty together, so candidates disadvantaged on MULTIPLE attributes
# are penalised more than any single attribute alone would predict.
combined_penalty = gender_pen * race_pen * religion_pen * continent_pen * age_pen

biased_score = (fair_score * combined_penalty + np.random.normal(0, 0.015, N)).clip(0, 1)

# ── STEP 4: SELECTION LABELS (top 15% hired, matching v1's convention) ──
selected_fair = (fair_score >= np.percentile(fair_score, 85)).astype(int)
selected_biased = (biased_score >= np.percentile(biased_score, 85)).astype(int)

df = pd.DataFrame({
    'ID': range(1, N + 1),
    'Gender': gender,
    'Age': age,
    'Race_Ethnicity': race_ethnicity,
    'Religion': religion,
    'Continent': continent,
    'Education': education,
    'Experience_Years': experience,
    'Interview_Score': interview_score,
    'Technical_Score': technical_score,
    'Communication_Score': communication_score,
    'Fair_Score': fair_score.round(4),
    'Biased_Score': biased_score.round(4),
    'Selected_Fair': selected_fair,
    'Selected': selected_biased,
})

df.to_csv('../data/hiring_dataset_v2.csv', index=False)
print(f"[1] Dataset v2 saved: hiring_dataset_v2.csv  ({N:,} rows, {df.shape[1]} columns)")

print("\n=== Overall selection rate ===")
print(f"  Fair (merit-only):  {df['Selected_Fair'].mean()*100:.1f}%")
print(f"  Biased (actual):    {df['Selected'].mean()*100:.1f}%")

for attr in ['Gender', 'Race_Ethnicity', 'Religion', 'Continent']:
    print(f"\n=== Selection rate by {attr} (biased/actual) ===")
    print((df.groupby(attr)['Selected'].mean() * 100).round(1).to_string())
