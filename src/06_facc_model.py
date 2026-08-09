"""
=============================================================================
FACC — Fairness-Aware Custom Classifier — built from scratch in NumPy
=============================================================================
Group D: Nexus Thinkers | University of Kashmir IT | Semester III, 2025

WHY FROM SCRATCH? This sandbox/development environment has no internet
access to install PyTorch or TensorFlow, so FACC's feature-relevance
mechanism — Q/K/V-style projections, scaled dot-product softmax
weighting, and the full backward pass — is implemented and hand-verified
below using only NumPy. The mechanism is built on the same core
mathematics used inside a Transformer's self-attention block (Vaswani
et al., "Attention Is All You Need", NeurIPS 2017), adapted here into a
custom classifier purpose-built for tabular HIRING features rather than
words (see dissertation Section 6.11 for the full theoretical grounding
and how FACC deliberately diverges from a standard Transformer).

WHY THIS MECHANISM FOR A BIAS PROJECT SPECIFICALLY?
Beyond being "one more algorithm" for the comparison table, FACC gives
us something the other five models can't: an interpretable per-feature
relevance weight. We can check, directly, whether the model is "paying
attention" to protected attributes (Gender, Race_Ethnicity, Religion,
Continent) as much as it weighs legitimate merit features
(Technical_Score, Experience_Years) — turning the model's own internal
weighting into a bias-detection tool.

Run:  python3 06_facc_model.py
=============================================================================
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from fairness_metrics_v2 import compute_group_metrics, four_fifths_check

RNG = np.random.default_rng(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class FairnessAwareCustomClassifier:
    """FACC: a single-head feature-wise custom classifier where every
    input feature is treated as a 'token', is weighted against every
    other feature via a scaled dot-product mechanism, and the resulting
    context vector feeds a small classification head."""

    def __init__(self, n_features, d_model=8, hidden=32, lr=2e-3, seed=42):
        rng = np.random.default_rng(seed)
        self.F, self.d = n_features, d_model
        g = lambda *shape: rng.normal(0, np.sqrt(2.0 / shape[0]), size=shape)

        self.We = g(n_features, d_model)
        self.Wq = g(d_model, d_model)
        self.Wk = g(d_model, d_model)
        self.Wv = g(d_model, d_model)
        self.Wo = g(n_features * d_model, hidden)
        self.bo = np.zeros(hidden)
        self.Wc = g(hidden, 1)
        self.bc = np.zeros(1)

        self.lr = lr
        self._adam_state = {}
        self.t = 0

    def _adam_init(self):
        for name in ['We', 'Wq', 'Wk', 'Wv', 'Wo', 'bo', 'Wc', 'bc']:
            p = getattr(self, name)
            self._adam_state[name] = {'m': np.zeros_like(p), 'v': np.zeros_like(p)}

    def forward(self, X):
        B, F = X.shape
        d = self.d
        E = X[:, :, None] * self.We[None, :, :]
        Q = E @ self.Wq; K = E @ self.Wk; V = E @ self.Wv
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d)
        A = softmax(scores, axis=-1)
        C = A @ V
        C_flat = C.reshape(B, F * d)
        Z1 = C_flat @ self.Wo + self.bo
        H1 = np.maximum(Z1, 0)
        Z2 = H1 @ self.Wc + self.bc
        y_hat = sigmoid(Z2).ravel()
        return y_hat, dict(X=X, E=E, Q=Q, K=K, V=V, A=A, C=C, C_flat=C_flat,
                            Z1=Z1, H1=H1, Z2=Z2, y_hat=y_hat)

    def backward(self, cache, y_true):
        B, F, d = cache['E'].shape
        y_hat = cache['y_hat']
        dZ2 = (y_hat - y_true).reshape(-1, 1) / B
        dWc = cache['H1'].T @ dZ2
        dbc = dZ2.sum(axis=0)
        dH1 = dZ2 @ self.Wc.T
        dZ1 = dH1 * (cache['Z1'] > 0)
        dWo = cache['C_flat'].T @ dZ1
        dbo = dZ1.sum(axis=0)
        dC_flat = dZ1 @ self.Wo.T
        dC = dC_flat.reshape(B, F, d)
        dA = dC @ cache['V'].transpose(0, 2, 1)
        dV = cache['A'].transpose(0, 2, 1) @ dC
        dscores = cache['A'] * (dA - np.sum(dA * cache['A'], axis=-1, keepdims=True))
        dscores = dscores / np.sqrt(d)
        dQ = dscores @ cache['K']
        dK = dscores.transpose(0, 2, 1) @ cache['Q']
        E = cache['E']
        dWq = np.einsum('bfd,bfe->de', E, dQ)
        dWk = np.einsum('bfd,bfe->de', E, dK)
        dWv = np.einsum('bfd,bfe->de', E, dV)
        dE = dQ @ self.Wq.T + dK @ self.Wk.T + dV @ self.Wv.T
        X = cache['X']
        dWe = np.einsum('bf,bfd->fd', X, dE)
        return dict(We=dWe, Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo, bo=dbo, Wc=dWc, bc=dbc)

    def adam_step(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        if not self._adam_state:
            self._adam_init()
        self.t += 1
        for name, grad in grads.items():
            s = self._adam_state[name]
            s['m'] = beta1 * s['m'] + (1 - beta1) * grad
            s['v'] = beta2 * s['v'] + (1 - beta2) * (grad ** 2)
            m_hat = s['m'] / (1 - beta1 ** self.t)
            v_hat = s['v'] / (1 - beta2 ** self.t)
            setattr(self, name, getattr(self, name) - self.lr * m_hat / (np.sqrt(v_hat) + eps))

    def fit(self, X, y, epochs=30, batch_size=256, verbose=True):
        n = X.shape[0]
        for epoch in range(epochs):
            perm = RNG.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]
                Xb, yb = X[idx], y[idx]
                y_hat, cache = self.forward(Xb)
                eps = 1e-9
                loss = -np.mean(yb * np.log(y_hat + eps) + (1 - yb) * np.log(1 - y_hat + eps))
                epoch_loss += loss * len(idx)
                self.adam_step(self.backward(cache, yb))
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(f"  epoch {epoch+1:3d}/{epochs}  loss={epoch_loss/n:.4f}")

    def predict_proba(self, X, batch_size=1024):
        probs = np.zeros(X.shape[0])
        for start in range(0, X.shape[0], batch_size):
            probs[start:start + batch_size], _ = self.forward(X[start:start + batch_size])
        return probs

    def mean_relevance_per_feature(self, X, batch_size=1024, sample=6000):
        if X.shape[0] > sample:
            idx = RNG.choice(X.shape[0], sample, replace=False)
            X = X[idx]
        totals = np.zeros(self.F); n_seen = 0
        for start in range(0, X.shape[0], batch_size):
            Xb = X[start:start + batch_size]
            _, cache = self.forward(Xb)
            totals += cache['A'].mean(axis=1).sum(axis=0)
            n_seen += Xb.shape[0]
        return totals / n_seen


def main():
    print("=" * 70)
    print("  FACC — FAIRNESS-AWARE CUSTOM CLASSIFIER (NumPy, from scratch)")
    print("=" * 70)

    df = pd.read_csv('../data/hiring_dataset_v2.csv')
    edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
    df['Edu_enc'] = df['Education'].map(edu_map)

    cat_cols = ['Gender', 'Race_Ethnicity', 'Religion', 'Continent']
    encoders = {c: LabelEncoder().fit(df[c]) for c in cat_cols}
    for c in cat_cols:
        df[f'{c}_enc'] = encoders[c].transform(df[c])

    num_features = ['Age', 'Experience_Years', 'Interview_Score',
                     'Technical_Score', 'Communication_Score', 'Edu_enc']
    cat_enc_features = [f'{c}_enc' for c in cat_cols]
    feature_names = num_features + cat_enc_features

    scaler = StandardScaler()
    X = df[feature_names].copy()
    X[num_features] = scaler.fit_transform(X[num_features])
    X = X.values.astype(float)
    y = df['Selected'].values.astype(float)
    sf = df['Selected_Fair'].values

    Xtr, Xte, ytr, yte, sf_tr, sf_te = train_test_split(
        X, y, sf, test_size=0.25, random_state=42, stratify=y)

    model = FairnessAwareCustomClassifier(n_features=X.shape[1], d_model=8, hidden=32, lr=2e-3, seed=42)
    print("\n[1] Training FACC (feature-wise custom classifier)...")
    model.fit(Xtr, ytr, epochs=30, batch_size=256)

    proba_test = model.predict_proba(Xte)
    pred_test = (proba_test >= 0.5).astype(int)
    metrics = {
        'accuracy': round(accuracy_score(yte, pred_test) * 100, 1),
        'precision': round(precision_score(yte, pred_test, zero_division=0) * 100, 1),
        'recall': round(recall_score(yte, pred_test, zero_division=0) * 100, 1),
        'f1': round(f1_score(yte, pred_test, zero_division=0) * 100, 1),
        'roc_auc': round(roc_auc_score(yte, proba_test), 3),
    }
    print(f"\n[2] Test metrics: {metrics}")

    print("\n[3] Fairness check (FACC vs 4 protected attributes):")
    proba_full = model.predict_proba(X)
    pred_full = (proba_full >= 0.5).astype(int)
    for attr in cat_cols:
        _, summary = compute_group_metrics(sf, pred_full, df[attr].values)
        print(f"    {attr:<15} DP diff={summary['demographic_parity_diff']:.3f}  "
              f"min DIR={summary['min_disparate_impact_ratio']:.3f} "
              f"[{four_fifths_check(summary['min_disparate_impact_ratio'])}]")

    print("\n[4] Mean relevance weight received per feature (top 10):")
    rel = model.mean_relevance_per_feature(X)
    rel_df = pd.DataFrame({'feature': feature_names, 'mean_relevance': rel})
    rel_df = rel_df.sort_values('mean_relevance', ascending=False)
    print(rel_df.head(10).to_string(index=False))

    protected_features = {f'{c}_enc' for c in cat_cols}
    protected_rank = rel_df[rel_df['feature'].isin(protected_features)]
    print(f"\n    -> Protected attributes occupy ranks "
          f"{[list(rel_df['feature']).index(f)+1 for f in protected_rank['feature']]} "
          f"out of {len(feature_names)} features by relevance weight received.")

    rel_df.to_csv('../data/attention_feature_weights_v2.csv', index=False)
    print("\nSaved: ../data/attention_feature_weights_v2.csv")


if __name__ == '__main__':
    main()
