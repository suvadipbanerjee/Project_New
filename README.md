# Evaluation Protocol Bias and Reproducibility in Machine Learning Models

This repository accompanies an empirical study investigating how different
model evaluation protocols influence reported performance and statistical
significance in machine learning experiments.

We compare:
- **Single-seed evaluation**
- **k-fold cross-validation**
- **Repeated cross-validation (multi-seed)**

across multiple datasets, tasks, models, and metrics, highlighting the risks of
optimistic reporting and poor reproducibility.


## Datasets & Tasks

| Dataset       | Task            | Metrics Used             |
|---------------|-----------------|--------------------------|
| Abalone       | Regression      | MSE, R², MAE             |
| Wine Quality  | Regression      | RMSE, R²                 |
| Diabetes      | Regression      | R²                       |
| Adult         | Classification  | Accuracy, F1, ROC-AUC    |

---

##  Evaluation Protocols

1. **Single Seed**  
   One fixed random seed; fast but unstable.

2. **Cross-Validation (CV)**  
   k-fold CV with one random split.

3. **Repeated Cross-Validation (Multi-Seed CV)**  
   k-fold CV repeated over multiple seeds to estimate mean and variance.

---

##  Statistical Analysis

To test whether single-seed and CV results differ significantly from repeated CV:

- **Paired t-test**
- **Wilcoxon signed-rank test**

Tests are applied **per dataset, model, and metric** using fold-level results.

---

## Key Visualizations

### 1. Evaluation Protocol Bias
Line plots showing how performance varies across evaluation protocols.

### 2. Statistical Significance Heatmaps
Heatmaps of p-values from t-tests and Wilcoxon tests.

### 3. Performance vs Reproducibility Trade-off
Scatter plot using the **Reproducibility Stability Index (RSI)**:
- Lower RSI → more stable model
- Higher RSI → higher variance across seeds

---

