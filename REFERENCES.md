# References — v2 Expansion

Consulted while designing the v2 dataset, the intersectional fairness
analysis, and the attention-based model. Cite these in the report's
literature review / bibliography chapter.

1. Fabris, A., Baranowska, N., Dennis, M. J., Graus, D., Hacker, P.,
   Saldivar, J., Zuiderveen Borgesius, F., & Biega, A. J. (2024).
   *Fairness and Bias in Algorithmic Hiring: A Multidisciplinary Survey.*
   ACM Transactions on Intelligent Systems and Technology.
   https://doi.org/10.1145/3696457
   — Used for: the taxonomy of protected attributes and "bias conducive
   factors" that motivated adding Race_Ethnicity, Religion, and Continent
   alongside Gender; the intersectionality framing in
   `07_intersectional_fairness.py`.

2. Wilson, K., & Caliskan, A. (2024). *Gender, Race, and Intersectional
   Bias in Resume Screening via Language Model Retrieval.*
   arXiv:2407.20371.
   — Used for: the specific claim that intersectional subgroups (e.g.
   Race x Gender) can face disadvantage larger than either attribute
   predicts alone — directly tested and reproduced in this project's
   intersectional analysis.

3. Kamiran, F., & Calders, T. (2012). *Data Preprocessing Techniques for
   Classification without Discrimination.* Knowledge and Information
   Systems, 33(1), 1–33.
   — Used for: the sample-reweighing mitigation technique in
   `08_mitigation_strategies.py`, generalised here from a single
   protected attribute to an intersectional group of four attributes.

4. Hardt, M., Price, E., & Srebro, N. (2016). *Equality of Opportunity in
   Supervised Learning.* Advances in Neural Information Processing
   Systems (NeurIPS) 29.
   — Used for: the Equal Opportunity / Equalized Odds definitions in
   `fairness_metrics_v2.py`.

5. Barocas, S., & Selbst, A. D. (2016). *Big Data's Disparate Impact.*
   California Law Review, 104(3), 671–732.
   — Used for: framing why a model can discriminate purely by learning
   from historically biased labels, without ever being told the
   protected attribute directly — the core mechanism this whole project
   simulates.

6. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez,
   A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You
   Need.* Advances in Neural Information Processing Systems (NeurIPS) 30.
   — Used for: the scaled dot-product self-attention mechanism
   (Q/K/V projections, softmax attention) implemented from scratch in
   `06_attention_model.py`.

7. US Equal Employment Opportunity Commission (1978). *Uniform Guidelines
   on Employee Selection Procedures* — the "four-fifths rule" (disparate
   impact ratio ≥ 0.80) used as the pass/fail threshold throughout the
   fairness dashboard.

---

**Note on originality**: the synthetic dataset, all Python code, and the
specific bias magnitudes/results reported are original work produced for
this project. The sources above are cited for the *concepts and metric
definitions* they establish (e.g. what "equalized odds" means, why
intersectional analysis matters), not for any data or code reused from
them.
