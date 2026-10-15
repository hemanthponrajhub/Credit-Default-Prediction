# Credit Card Default Prediction

Predict which credit card customers will miss next month's payment, so a bank's risk team can act early (a reminder call, a payment plan, a limit review).

**Author:** Hemanth Ponraj · M.S. Data Science & Analytics, Georgia State University

---

## Results at a glance

| Metric | Value |
|---|---|
| Best model | Gradient Boosting |
| CV ROC-AUC (5-fold) | **0.785** |
| Test ROC-AUC | **0.782** |
| Test PR-AUC | 0.557 (baseline 0.221) |
| Defaulters caught at 0.50 cutoff | 36% (precision 67%) |
| Defaulters caught at tuned 0.25 cutoff | **58%** (precision ~49%) |

Test score is close to the CV score, so the model is not overfitting.

---

## Dataset

[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) (Taiwan, 2005)

- 30,000 customers, 25 columns
- 22.1% defaulted
- No missing values
- Columns: credit limit, age, education, 6 months of repayment status, bill amounts and payment amounts

---

## Approach

1. **Clean** – fold undocumented EDUCATION / MARRIAGE codes into "Other", drop ID, rename `PAY_0` → `PAY_1`.
2. **Engineer features**
   - `MONTHS_LATE` – number of late months in the last 6
   - `UTILIZATION` – latest bill ÷ credit limit
   - `PAY_RATIO` – total paid ÷ total billed (capped 0–1)
3. **Explore** – customers 2+ months late last month default ~70% of the time vs ~13% for on-time payers.
4. **Fairness by design** – SEX and MARRIAGE are kept **out** of the model (US Equal Credit Opportunity Act). They are only used to check fairness at the end.
5. **Model** – 6 models in one scikit-learn Pipeline, compared with stratified 5-fold CV on ROC-AUC.
6. **Tune the threshold** – pick the highest cutoff that still catches 60% of defaulters, using out-of-fold predictions on training data only.
7. **Test once** – score the held-out 20% test set a single time.

### Model comparison (CV ROC-AUC)

| Model | ROC-AUC |
|---|---|
| **Gradient Boosting** | **0.785** |
| Voting (soft) | 0.782 |
| Random Forest | 0.763 |
| Logistic Regression | 0.757 |
| Naive Bayes | 0.731 |
| Decision Tree | 0.612 |

---

## Key drivers of default

1. Last month's repayment status (`PAY_1`)
2. Number of late months (`MONTHS_LATE`)
3. Utilization
4. Latest bill amount
5. Share of bill paid back
6. Credit limit

## Fairness check (test set, tuned cutoff)

| Group | Flag rate | Recall |
|---|---|---|
| Female | 25% | 0.58 |
| Male | 29% | 0.60 |

Recall is within 2 points across groups.

---

## Business takeaways

- **Act on last month first.** A customer who just went late is the highest-risk group.
- **Watch the pattern.** Repeated late months and low payback are early warning signs.
- **Use the ranked list.** Work from the top down and move the cutoff as costs change.

---

## Project structure

```
├── Credit_Card_Default_Prediction.ipynb   # main notebook
├── Credit_Card_Default_Prediction.html    # notebook export (view without running)
├── default of credit card clients.xls     # dataset
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/<your-username>/credit-card-default-prediction.git
cd credit-card-default-prediction
pip install -r requirements.txt
jupyter notebook Credit_Card_Default_Prediction.ipynb
```

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## Next steps

- Tune Gradient Boosting hyperparameters inside CV
- Time-based validation (train on older months, test on newer)
- Set the cutoff using real call and default costs
