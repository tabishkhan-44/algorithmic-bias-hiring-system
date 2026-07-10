"""
Fairness Metrics Module v2 — generalized to ANY protected attribute
(not just Gender), plus intersectional (multi-attribute) analysis.

Extends fairness_metrics.py (v1, gender-only) so v1's module and API stay
untouched and backward compatible. Import this module when you need to
check fairness across Race_Ethnicity, Religion, Continent, or Age bands,
or across a combination of them.

Usage:
    from fairness_metrics_v2 import compute_group_metrics, four_fifths_check

References for metric definitions:
  - Disparate Impact / "four-fifths rule": US EEOC Uniform Guidelines
    on Employee Selection Procedures (1978).
  - Statistical Parity Difference, Equalized Odds: Hardt, Price & Srebro,
    "Equality of Opportunity in Supervised Learning", NeurIPS 2016.
  - Reweighing for bias mitigation: Kamiran & Calders, "Data
    preprocessing techniques for classification without discrimination",
    Knowledge and Information Systems, 2012.
"""
import numpy as np
import pandas as pd


def selection_rate(y_pred, group_mask):
    return y_pred[group_mask].mean() if group_mask.sum() > 0 else np.nan


def true_positive_rate(y_true, y_pred, group_mask):
    mask = group_mask & (y_true == 1)
    return y_pred[mask].mean() if mask.sum() > 0 else np.nan


def false_positive_rate(y_true, y_pred, group_mask):
    mask = group_mask & (y_true == 0)
    return y_pred[mask].mean() if mask.sum() > 0 else np.nan


def compute_group_metrics(y_true, y_pred, group_arr, privileged_group=None):
    """Selection rate, TPR, FPR for every category of ONE protected
    attribute, plus the overall demographic parity diff / disparate
    impact ratio / equalized-odds diff relative to whichever group has
    the HIGHEST selection rate (the de facto privileged group), unless
    privileged_group is given explicitly."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    group_arr = np.asarray(group_arr)
    categories = pd.unique(group_arr)

    rows = []
    for cat in categories:
        mask = group_arr == cat
        rows.append({
            'group': cat,
            'n': int(mask.sum()),
            'selection_rate': selection_rate(y_pred, mask),
            'tpr': true_positive_rate(y_true, y_pred, mask),
            'fpr': false_positive_rate(y_true, y_pred, mask),
        })
    table = pd.DataFrame(rows).set_index('group')

    priv = privileged_group or table['selection_rate'].idxmax()
    sr_priv = table.loc[priv, 'selection_rate']

    table['disparate_impact_ratio'] = table['selection_rate'] / sr_priv
    table['statistical_parity_diff'] = sr_priv - table['selection_rate']
    table['equal_opportunity_diff'] = table.loc[priv, 'tpr'] - table['tpr']

    summary = {
        'privileged_group': priv,
        'demographic_parity_diff': round(table['statistical_parity_diff'].max(), 4),
        'min_disparate_impact_ratio': round(table['disparate_impact_ratio'].min(), 4),
        'max_equal_opportunity_diff': round(table['equal_opportunity_diff'].abs().max(), 4),
        'passes_four_fifths_rule': bool(table['disparate_impact_ratio'].min() >= 0.8),
    }
    return table.round(4), summary


def four_fifths_check(ratio):
    return "PASS" if ratio >= 0.8 else "FAIL (< 0.80 legal minimum)"


def compute_intersectional_metrics(y_true, y_pred, attr1_arr, attr2_arr,
                                     attr1_name="attr1", attr2_name="attr2",
                                     min_group_size=30):
    """Combines two protected attributes into intersectional groups
    (e.g. Race x Gender = 'Black_Female') and reports selection rate for
    each combination that has enough samples to be statistically
    meaningful. This is the analysis explicitly called for by Wilson &
    Caliskan (2024) and the Fabris et al. (2024) survey: aggregate,
    single-attribute fairness metrics can look acceptable while masking
    much larger gaps for people at the intersection of two disadvantaged
    groups (e.g. a candidate who is both an ethnic minority AND a
    religious minority)."""
    df = pd.DataFrame({
        attr1_name: attr1_arr, attr2_name: attr2_arr,
        'y_true': y_true, 'y_pred': y_pred,
    })
    df['intersection'] = df[attr1_name].astype(str) + " + " + df[attr2_name].astype(str)

    rows = []
    for group, sub in df.groupby('intersection'):
        if len(sub) < min_group_size:
            continue
        rows.append({
            'intersection': group,
            'n': len(sub),
            'selection_rate': round(sub['y_pred'].mean(), 4),
        })
    table = pd.DataFrame(rows).sort_values('selection_rate')
    if len(table) == 0:
        return table, {}

    best = table.iloc[-1]
    worst = table.iloc[0]
    summary = {
        'most_disadvantaged_intersection': worst['intersection'],
        'most_disadvantaged_rate': worst['selection_rate'],
        'most_advantaged_intersection': best['intersection'],
        'most_advantaged_rate': best['selection_rate'],
        'intersectional_gap': round(best['selection_rate'] - worst['selection_rate'], 4),
    }
    return table.reset_index(drop=True), summary


if __name__ == '__main__':
    df = pd.read_csv('../data/hiring_dataset_v2.csv')
    print("=" * 70)
    print("  FAIRNESS METRICS v2 — quick self-test on the ground-truth labels")
    print("=" * 70)
    for attr in ['Gender', 'Race_Ethnicity', 'Religion', 'Continent']:
        table, summary = compute_group_metrics(
            df['Selected_Fair'].values, df['Selected'].values, df[attr].values)
        print(f"\n--- {attr} ---")
        print(table[['n', 'selection_rate', 'disparate_impact_ratio',
                      'statistical_parity_diff', 'equal_opportunity_diff']])
        print(f"  Demographic parity diff : {summary['demographic_parity_diff']}")
        print(f"  Min disparate impact    : {summary['min_disparate_impact_ratio']} "
              f"[{four_fifths_check(summary['min_disparate_impact_ratio'])}]")

    print("\n--- Intersectional: Race_Ethnicity x Gender ---")
    itable, isummary = compute_intersectional_metrics(
        df['Selected_Fair'].values, df['Selected'].values,
        df['Race_Ethnicity'].values, df['Gender'].values,
        'Race_Ethnicity', 'Gender')
    print(itable.to_string(index=False))
    print(f"\n  Most disadvantaged: {isummary['most_disadvantaged_intersection']} "
          f"({isummary['most_disadvantaged_rate']*100:.1f}%)")
    print(f"  Most advantaged   : {isummary['most_advantaged_intersection']} "
          f"({isummary['most_advantaged_rate']*100:.1f}%)")
    print(f"  Intersectional gap: {isummary['intersectional_gap']*100:.1f} percentage points")
