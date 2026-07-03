"""
=======================================================================
Algorithmic Bias in Everyday Systems — Main Pipeline
=======================================================================
Group D: Nexus Thinkers | Univ. of Kashmir IT | Semester III, 2025
Run: python3 main_pipeline.py
=======================================================================
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

# ── STEP 1: GENERATE DATA ───────────────────────────────────────────────
np.random.seed(42); N = 1000
gender = np.random.choice(['Male','Female','Non-Binary'], N, p=[0.60,0.35,0.05])
age    = np.where(gender=='Male',
           np.random.normal(29,4,N).clip(22,55).astype(int),
           np.random.normal(28,4,N).clip(22,55).astype(int))
education = np.where(gender=='Male',
    np.random.choice(['High School','Bachelor','Master','PhD'],N,p=[0.10,0.50,0.32,0.08]),
    np.random.choice(['High School','Bachelor','Master','PhD'],N,p=[0.08,0.48,0.36,0.08]))
experience = np.where(gender=='Male',
    np.random.gamma(3.0,1.5,N).clip(0,15).round(1),
    np.random.gamma(2.5,1.4,N).clip(0,15).round(1))
interview_score    = np.where(gender=='Male',
    np.random.normal(68,12,N), np.random.normal(70,12,N)).clip(20,100).round(1)
technical_score    = np.where(gender=='Male',
    np.random.normal(70,14,N),
    np.random.normal(67,15,N)+np.random.normal(0,8,N)).clip(20,100).round(1)
communication_score= np.where(gender=='Male',
    np.random.normal(66,13,N), np.random.normal(70,12,N)).clip(20,100).round(1)

edu_map = {'High School':0,'Bachelor':1,'Master':2,'PhD':3}
edu_num = np.array([edu_map[e] for e in education])

def normalize(arr): return (arr-arr.min())/(arr.max()-arr.min())

fair_score = (0.25*normalize(edu_num) + 0.20*normalize(experience) +
              0.20*normalize(interview_score) + 0.25*normalize(technical_score) +
              0.10*normalize(communication_score))

bias_penalty = np.where(gender=='Female',0.88,np.where(gender=='Non-Binary',0.93,1.0))
biased_score = (fair_score * bias_penalty + np.random.normal(0,0.015,N)).clip(0,1)

selected_fair   = (fair_score   >= np.percentile(fair_score,   85)).astype(int)
selected_biased = (biased_score >= np.percentile(biased_score, 85)).astype(int)

df = pd.DataFrame({'ID':range(1,N+1),'Gender':gender,'Age':age,'Education':education,
    'Experience_Years':experience,'Interview_Score':interview_score,
    'Technical_Score':technical_score,'Communication_Score':communication_score,
    'Fair_Score':fair_score.round(4),'Biased_Score':biased_score.round(4),
    'Selected_Fair':selected_fair,'Selected':selected_biased})

df.to_csv('hiring_dataset.csv', index=False)
print("[1] Dataset saved: hiring_dataset.csv")

# ── STEP 2: TRAIN MODEL ─────────────────────────────────────────────────
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

le = LabelEncoder()
df['Gender_enc'] = le.fit_transform(df['Gender'])
df['Edu_enc']    = df['Education'].map(edu_map)
feats = ['Gender_enc','Age','Edu_enc','Experience_Years','Interview_Score','Technical_Score','Communication_Score']
X = df[feats].values; y = df['Selected'].values
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
clf = DecisionTreeClassifier(max_depth=6, random_state=42)
clf.fit(Xtr, ytr)
yp = clf.predict(Xte); yprob = clf.predict_proba(Xte)[:,1]

print("[2] Model trained: Decision Tree (max_depth=6)")

# ── STEP 3: METRICS ─────────────────────────────────────────────────────
print("\n=== Classification Metrics ===")
print(f"  Accuracy  : {accuracy_score(yte,yp)*100:.1f}%")
print(f"  Precision : {precision_score(yte,yp)*100:.1f}%")
print(f"  Recall    : {recall_score(yte,yp)*100:.1f}%")
print(f"  F1 Score  : {f1_score(yte,yp)*100:.1f}%")
print(f"  AUC-ROC   : {roc_auc_score(yte,yprob):.3f}")
print("\n  Confusion Matrix:")
cm = confusion_matrix(yte, yp)
print(f"  [[TN={cm[0,0]}  FP={cm[0,1]}],")
print(f"   [FN={cm[1,0]}  TP={cm[1,1]}]]")

# ── STEP 4: FAIRNESS METRICS ────────────────────────────────────────────
print("\n=== Fairness Metrics ===")
for g in ['Male','Female','Non-Binary']:
    sub = df[df.Gender==g]; sr = sub['Selected'].mean()*100
    print(f"  {g:12s}: Selection Rate = {sr:.1f}%")
msr = df[df.Gender=='Male']['Selected'].mean()
fsr = df[df.Gender=='Female']['Selected'].mean()
print(f"\n  Disparate Impact Ratio  : {fsr/msr:.3f}  (legal min: 0.80)")
print(f"  Statistical Parity Diff : {msr-fsr:.3f}  (ideal: 0.000)")
tpr_m = (df[(df.Gender=='Male')&(df.Selected_Fair==1)]['Selected']).mean()
tpr_f = (df[(df.Gender=='Female')&(df.Selected_Fair==1)]['Selected']).mean()
print(f"  True Positive Rate (M)  : {tpr_m:.3f}")
print(f"  True Positive Rate (F)  : {tpr_f:.3f}")
print(f"  Equal Opportunity Diff  : {tpr_m-tpr_f:.3f}  (ideal: 0.000)")
print("\n[3] Fairness evaluation complete.")
print("\nRun 02_eda.py for visualizations → All figures saved to ./graphs/")
