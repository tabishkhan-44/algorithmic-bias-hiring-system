"""
=============================================================================
Bias Mitigation Strategies v2
=============================================================================
Group D: Nexus Thinkers | University of Kashmir IT | Semester III, 2025

Applies two mitigation techniques to XGBoost (chosen because Section 5's
comparison shows it as both the most accurate AND the most biased model —
the clearest demonstration case) and re-measures fairness on all four
protected attributes.

  1. SAMPLE REWEIGHTING (Kamiran & Calders, 2012) — generalised here to an
     INTERSECTIONAL group formed by combining all four protected
     attributes at once, so the reweighting doesn't just fix Gender while
     leaving Race/Religion/Continent bias untouched.
  2. GROUP-SPECIFIC DECISION THRESHOLDS (post-processing) — instead of one
     0.5 cutoff for everyone, each group gets its own cutoff so its
     selection rate matches the overall target rate.

CAUTION (flagged explicitly, not hidden): reweighting can overcorrect for
small subgroups (e.g. Non-Binary, Jewish, Oceania — all under ~1,600
rows), sometimes flipping the bias direction. This script reports that
rather than cherry-picking only the favourable results.

Run:  python3 08_mitigation_strategies.py
=============================================================================
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

np.random.seed(42)
MAROON = '#800000'; GOLD = '#C8A000'; GREEN = '#2E7D32'; DPI = 96
PROTECTED_ATTRS = ['Gender', 'Race_Ethnicity', 'Religion', 'Continent']


def compute_reweighing_weights(df_train, y_train, protected_attrs):
    """Kamiran & Calders (2012) reweighing, generalised to an
    intersectional group combining several protected attributes."""
    d = df_train.copy()
    d['_label'] = y_train
    d['_group'] = d[protected_attrs].astype(str).agg('_'.join, axis=1)
    n = len(d)
    weights = np.ones(n)
    for group in d['_group'].unique():
        for label in [0, 1]:
            mask = (d['_group'] == group) & (d['_label'] == label)
            n_gl = mask.sum()
            if n_gl == 0:
                continue
            p_group = (d['_group'] == group).mean()
            p_label = (d['_label'] == label).mean()
            weights[mask.values] = (p_group * p_label * n) / n_gl
    return weights


def group_specific_thresholds(proba, group_labels, target_rate):
    groups = pd.Series(group_labels)
    adjusted = np.zeros(len(proba), dtype=int)
    thresholds = {}
    for group in groups.unique():
        mask = (groups == group).values
        gp = proba[mask]
        if len(gp) == 0:
            continue
        q = np.clip(1 - target_rate, 0, 1)
        thr = float(np.quantile(gp, q))
        thresholds[group] = round(thr, 4)
        adjusted[mask] = (gp >= thr).astype(int)
    return adjusted, thresholds


def dp_diff(pred, group_labels):
    d = pd.DataFrame({'pred': pred, 'group': group_labels})
    rates = d.groupby('group', observed=True)['pred'].mean()
    return round(rates.max() - rates.min(), 4)


def main():
    print("=" * 70)
    print("  BIAS MITIGATION STRATEGIES v2 (XGBoost)")
    print("=" * 70)

    df = pd.read_csv('../data/hiring_dataset_v2.csv')
    edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
    df['Edu_enc'] = df['Education'].map(edu_map)
    for c in PROTECTED_ATTRS:
        df[f'{c}_enc'] = LabelEncoder().fit_transform(df[c])

    features = ['Gender_enc', 'Age', 'Race_Ethnicity_enc', 'Religion_enc',
                'Continent_enc', 'Edu_enc', 'Experience_Years',
                'Interview_Score', 'Technical_Score', 'Communication_Score']
    X = df[features].values
    y = df['Selected'].values

    idx = np.arange(len(df))
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(
        X, y, idx, test_size=0.25, random_state=42, stratify=y)
    df_train, df_test = df.iloc[itr].reset_index(drop=True), df.iloc[ite].reset_index(drop=True)

    # ---- 1. Baseline ------------------------------------------------------
    base = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                          random_state=42, eval_metric='logloss', verbosity=0)
    base.fit(Xtr, ytr)
    base_proba = base.predict_proba(Xte)[:, 1]
    base_pred = (base_proba >= 0.5).astype(int)

    base_dp = {a: dp_diff(base_pred, df_test[a]) for a in PROTECTED_ATTRS}
    base_acc = round(accuracy_score(yte, base_pred) * 100, 1)
    base_f1 = round(f1_score(yte, base_pred, zero_division=0) * 100, 1)

    print(f"\n[1] BASELINE (no mitigation): Accuracy={base_acc}%  F1={base_f1}%")
    for a in PROTECTED_ATTRS:
        print(f"    {a:<15} DP diff = {base_dp[a]}")

    # ---- 2. Reweighting -----------------------------------------------------
    weights = compute_reweighing_weights(df_train, ytr, PROTECTED_ATTRS)
    rw = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                        random_state=42, eval_metric='logloss', verbosity=0)
    rw.fit(Xtr, ytr, sample_weight=weights)
    rw_proba = rw.predict_proba(Xte)[:, 1]
    rw_pred = (rw_proba >= 0.5).astype(int)

    rw_dp = {a: dp_diff(rw_pred, df_test[a]) for a in PROTECTED_ATTRS}
    rw_acc = round(accuracy_score(yte, rw_pred) * 100, 1)
    rw_f1 = round(f1_score(yte, rw_pred, zero_division=0) * 100, 1)

    print(f"\n[2] INTERSECTIONAL REWEIGHTING: Accuracy={rw_acc}%  F1={rw_f1}%")
    for a in PROTECTED_ATTRS:
        change = "better" if rw_dp[a] < base_dp[a] else "WORSE (overcorrected)"
        print(f"    {a:<15} DP diff = {rw_dp[a]}  ({change})")

    # ---- 3. Group-specific thresholds ---------------------------------------
    target_rate = yte.mean()
    thr_dp = {}
    thr_details = {}
    for a in PROTECTED_ATTRS:
        adj_pred, thresholds = group_specific_thresholds(base_proba, df_test[a], target_rate)
        thr_dp[a] = dp_diff(adj_pred, df_test[a])
        thr_details[a] = thresholds

    print("\n[3] GROUP-SPECIFIC THRESHOLDS (post-processing on baseline model):")
    for a in PROTECTED_ATTRS:
        print(f"    {a:<15} DP diff after adjustment = {thr_dp[a]}")

    print("""
[4] CAUTION: reweighting reduces demographic parity difference for most
    attributes but note the following, reported without cherry-picking:
    small subgroups (Non-Binary n~1,000, Jewish n~1,300, Oceania n~1,000)
    can show unstable / overcorrected results because reweighting relies
    on having enough samples per group x label cell to estimate a stable
    weight. Group-specific thresholds do not have this weakness because
    they only need enough samples to estimate one quantile per group.
""")

    # ---- Figure: mitigation comparison ---------------------------------------
    x = np.arange(len(PROTECTED_ATTRS)); w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5), dpi=DPI)
    ax.bar(x - w, [base_dp[a] for a in PROTECTED_ATTRS], w, label='Baseline', color='#C62828')
    ax.bar(x, [rw_dp[a] for a in PROTECTED_ATTRS], w, label='Reweighted', color=GOLD)
    ax.bar(x + w, [thr_dp[a] for a in PROTECTED_ATTRS], w, label='Group-specific thresholds', color=GREEN)
    ax.set_xticks(x); ax.set_xticklabels(PROTECTED_ATTRS)
    ax.set_ylabel('Demographic Parity Difference')
    ax.set_title('Figure 23: Bias Mitigation Impact on XGBoost (lower = fairer)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('../graphs/fig23_mitigation_comparison.png', dpi=DPI, facecolor='white')
    plt.close()
    print("Saved: ../graphs/fig23_mitigation_comparison.png")

    # ---- Save results table ---------------------------------------------------
    rows = []
    for a in PROTECTED_ATTRS:
        rows.append({'attribute': a, 'baseline_dp_diff': base_dp[a],
                      'reweighted_dp_diff': rw_dp[a], 'threshold_adjusted_dp_diff': thr_dp[a]})
    pd.DataFrame(rows).to_csv('../data/mitigation_results_v2.csv', index=False)
    print("Saved: ../data/mitigation_results_v2.csv")


if __name__ == '__main__':
    main()
