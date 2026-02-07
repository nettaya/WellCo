# Uplift Modeling for Churn Outreach Optimization

## 1. Project Goal
This project predicts which members are most likely to benefit from outreach so that outreach budget is allocated where it has the highest incremental effect on reducing churn.

The assignment includes a treatment signal (`outreach`) that occurs between the observation window and churn measurement. The modeling approach explicitly uses this signal to estimate **incremental treatment effect (uplift)** rather than only churn risk.

## 2. Repository Structure
- [`preprocessing.py`](./preprocessing.py): builds train/test feature tables from raw sources.
- [`model.py`](./model.py): trains churn baselines, trains uplift model, evaluates uplift, selects outreach size `n`, and writes outputs.
- [`raw_data/`](./raw_data/): source files.
- [`preprocessed_data/`](./preprocessed_data/): merged feature tables ([`train.csv`](./preprocessed_data/train.csv), [`test.csv`](./preprocessed_data/test.csv)).
- [`outputs/`](./outputs/): model outputs and decision artifacts.
- [`eda.ipynb`](./eda.ipynb): optional exploratory analysis notebook.

## 3. Data & Feature Engineering

### 3.1 Data sources
- `churn_labels.csv`: member anchors + target (`churn`) + treatment (`outreach`).
- `claims.csv`: ICD-10 clinical events.
- `web_visits.csv`: educational content engagement.
- `app_usage.csv`: session-based activity.

### 3.2 Feature construction (in [`preprocessing.py`](./preprocessing.py))
Features are built per member and merged into a single modeling table:

- Member profile:
  - `membership_days`
- Clinical burden:
  - `flag_diabetes` (E11.9), `flag_hypertension` (I10), `flag_dietary_counseling` (Z71.3)
  - `claims_count`, `claims_unique_icd_count`, `critical_icd_count`
- Content engagement topics:
  - topic scores (`topic_nutrition`, `topic_heart_bp`, `topic_diabetes`, etc.)
  - `web_visits_total`, `topic_diversity`
- App behavior:
  - `total_sessions`, `avg_hours_between_sessions`, `avg_sessions_per_user`, `highly_active_user`

### 3.3 Feature selection rationale
In [`model.py`](./model.py), feature selection is intentionally simple and robust:
- Use columns present in both train and test.
- Exclude `member_id`, `churn`, `outreach` from predictors.
- Keep numeric features only.

Why this is reasonable:
- **Domain relevance**: selected features capture health risk proxies and engagement behavior linked to retention.
- **Data quality**: numeric-only and train/test intersection reduce schema mismatch and encoding risk.
- **Predictive utility**: mix of clinical + behavioral + content signals supports both churn prediction and heterogeneous treatment response.

## 4. Modeling Approach

### 4.1 Churn baseline models
Three classifiers are compared for churn discrimination:
- Logistic Regression (with scaling + class balancing)
- Gradient Boosting
- Random Forest

Purpose: establish baseline churn predictiveness and sanity-check feature quality.

### 4.2 Uplift model (using outreach in modeling)
A **T-learner** is used:
- Train model `M0` on members with `outreach=0` to estimate `P(churn | no outreach, X)`.
- Train model `M1` on members with `outreach=1` to estimate `P(churn | outreach, X)`.
- Compute uplift score:

$$
\mathrm{uplift\space reduction}(x) = p_0(x) - p_1(x)
$$

Positive score means outreach is expected to reduce churn for that member.

This explicitly incorporates the outreach event as treatment and directly answers the business question: *who benefits most from outreach?*

## 5. Evaluation Strategy

### 5.1 Churn model evaluation
Current configuration uses 5-fold stratified CV (`USE_CV=True`) and reports ROC-AUC mean/std.

Latest results (`outputs/churn_cv_metrics.csv`):
- Logistic Regression: `0.6699 ± 0.0189`
- Gradient Boosting: `0.6551 ± 0.0193`
- Random Forest: `0.6276 ± 0.0183`

### 5.2 Uplift evaluation
Uplift is evaluated on a validation split stratified by `(churn, outreach)`:
- **Uplift Gain Curve**: cumulative incremental gain when targeting top-ranked members.
- **AUUC** (Area Under Uplift Curve): summary ranking metric.
- **Random baseline line** for comparison.

Latest AUUC (`outputs/validation_auuc_score.csv`):
- `13.6288` (positive, indicating useful uplift ranking signal)

Why these metrics:
- ROC-AUC is suitable for pure churn discrimination.
- AUUC/gain are suitable for treatment ranking quality.
- Net-based targeting is suitable for decision-making under outreach cost.

## 6. Selecting Outreach Size `n`

### 6.1 Decision objective
For sorted uplift scores \(u_{(i)}\), choose `n` that maximizes:

$$
\mathrm{Net}(n) = \sum_{i=1}^{n} u_{(i)} - c \cdot n
$$

Where:
- `c` = outreach cost per contacted member (as a fraction of value)

### 6.2 Why not cost-only?
Cost is the explicit optimization variable in code, but operationally `n` should also consider:
- outreach team capacity,
- fairness / segment coverage constraints,
- contact fatigue and policy limits,
- uncertainty bands (stability over time).

### 6.3 Current sensitivity results
From `outputs/validation_optimal_n_by_cost.csv`:
- `c=0.005 -> best_n=1518`
- `c=0.01  -> best_n=1425`
- `c=0.02  -> best_n=1197`
- `c=0.03  -> best_n=945`
- `c=0.05  -> best_n=492`

As expected: higher cost leads to smaller optimal outreach size.

## 7. Outputs and How to Read Them
Main files generated by `model.py`:

- `outputs/churn_cv_metrics.csv`
  - Churn CV benchmark metrics.
- `outputs/validation_uplift_gain_curve.csv`
  - `top_n`, `fraction`, `gain`, `random_gain` for uplift curve plotting.
- `outputs/validation_auuc_score.csv`
  - Single AUUC summary value.
- `outputs/validation_optimal_n_by_cost.csv`
  - Best outreach size for each tested cost.
- `outputs/validation_cumulative_net_curve_c_0.01.csv`
  - Cumulative gain/net across `n` for representative cost.
- `outputs/validation_ranked_uplift_cumsum_c_0.01.csv`
  - Ranked validation uplift and cumulative sums.
- `outputs/test_users_ranked_by_predicted_uplift.csv`
  - Final test ranking with member-level scores.
- `outputs/test_recommended_targets_topn_c_0.01.csv`
  - Recommended contact list for selected cost.
- `outputs/test_cumulative_uplift_and_net_c_0.01.csv`
  - Test cumulative uplift/net trajectory.

## Environment
- Python 3.x
- Dependencies in `requirements.txt`

Run end-to-end:

```bash
python3 preprocessing.py
python3 model.py
```


Outputs will be written to `outputs/`.
