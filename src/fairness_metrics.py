"""
Fairness Metrics Module — reusable functions for bias analysis.
Usage: from fairness_metrics import compute_all_metrics
"""
import numpy as np

def selection_rate(y_pred, group_mask):
    return y_pred[group_mask].mean()

def disparate_impact_ratio(y_pred, privileged_mask, unprivileged_mask):
    sr_priv   = selection_rate(y_pred, privileged_mask)
    sr_unpriv = selection_rate(y_pred, unprivileged_mask)
    return sr_unpriv / sr_priv if sr_priv > 0 else 0.0

def statistical_parity_difference(y_pred, privileged_mask, unprivileged_mask):
    return (selection_rate(y_pred, privileged_mask) -
            selection_rate(y_pred, unprivileged_mask))

def true_positive_rate(y_true, y_pred, group_mask):
    pos_mask = (y_true == 1) & group_mask
    return y_pred[pos_mask].mean() if pos_mask.sum() > 0 else 0.0

def false_positive_rate(y_true, y_pred, group_mask):
    neg_mask = (y_true == 0) & group_mask
    return y_pred[neg_mask].mean() if neg_mask.sum() > 0 else 0.0

def false_negative_rate(y_true, y_pred, group_mask):
    return 1 - true_positive_rate(y_true, y_pred, group_mask)

def equalized_odds_difference(y_true, y_pred, priv_mask, unpriv_mask):
    tpr_diff = (true_positive_rate(y_true,y_pred,priv_mask) -
                true_positive_rate(y_true,y_pred,unpriv_mask))
    fpr_diff = (false_positive_rate(y_true,y_pred,priv_mask) -
                false_positive_rate(y_true,y_pred,unpriv_mask))
    return max(abs(tpr_diff), abs(fpr_diff))

def positive_predictive_value(y_true, y_pred, group_mask):
    sel_mask = (y_pred == 1) & group_mask
    return y_true[sel_mask].mean() if sel_mask.sum() > 0 else 0.0

def compute_all_metrics(y_true, y_pred, gender_arr):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred).round().astype(int)
    gender_arr=np.asarray(gender_arr)
    m_mask=(gender_arr=='Male'); f_mask=(gender_arr=='Female')
    results={
        'Selection Rate (Male)'  : selection_rate(y_pred, m_mask)*100,
        'Selection Rate (Female)': selection_rate(y_pred, f_mask)*100,
        'Disparate Impact Ratio' : disparate_impact_ratio(y_pred,m_mask,f_mask),
        'Stat. Parity Difference': statistical_parity_difference(y_pred,m_mask,f_mask),
        'TPR (Male)'             : true_positive_rate(y_true,y_pred,m_mask),
        'TPR (Female)'           : true_positive_rate(y_true,y_pred,f_mask),
        'FPR (Male)'             : false_positive_rate(y_true,y_pred,m_mask),
        'FPR (Female)'           : false_positive_rate(y_true,y_pred,f_mask),
        'FNR (Male)'             : false_negative_rate(y_true,y_pred,m_mask),
        'FNR (Female)'           : false_negative_rate(y_true,y_pred,f_mask),
        'PPV (Male)'             : positive_predictive_value(y_true,y_pred,m_mask),
        'PPV (Female)'           : positive_predictive_value(y_true,y_pred,f_mask),
        'EO Difference'          : equalized_odds_difference(y_true,y_pred,m_mask,f_mask),
    }
    print("="*55)
    print("  FAIRNESS METRICS REPORT")
    print("="*55)
    for k,v in results.items():
        unit='%' if 'Rate' in k and 'Ratio' not in k and 'Differ' not in k else ''
        val=f"{v:.1f}{unit}" if '%' in unit else f"{v:.3f}"
        flag=" ⚠  VIOLATION" if k=='Disparate Impact Ratio' and v<0.8 else ""
        print(f"  {k:32s}: {val}{flag}")
    return results

if __name__=='__main__':
    import pandas as pd
    df=pd.read_csv('hiring_dataset.csv')
    compute_all_metrics(df['Selected_Fair'].values, df['Selected'].values, df['Gender'].values)
