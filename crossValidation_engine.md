**Purpose**
Run quick, reproducible cross-validation for our two tasks:

* **RQ1 (regression):** predict `final_perf` from run metadata/hyperparameters.
  Models: `LinearRegression`, `RandomForestRegressor`.
* **RQ2 (classification):** predict `likely_by_horizon_conservative` (0/1).
  Models: `LogisticRegression(class_weight="balanced")`, `RandomForestClassifier(class_weight="balanced")`.

---

## Requirements

* Python **3.10** (3.10.12 recommended)
* `pip install pandas numpy scikit-learn`
* (Optional) `matplotlib` is imported but not required for console runs.

---

## Expected input

Place a CSV named **`DataSet_complete.csv`** in the same directory as the script (or adjust the path in the code).

**Required columns**

| Column                           | Type / Notes                                  |
| -------------------------------- | --------------------------------------------- |
| `env_name`                       | string/categorical (e.g., `Worm`, `Pyramids`) |
| `learning_rate`                  | numeric                                       |
| `batch_size`                     | numeric (int)                                 |
| `nn_arch_depth`                  | numeric (int)                                 |
| `cpu_cores_logical`              | numeric (int)                                 |
| `ram_total_gb`                   | numeric (float)                               |
| `final_perf`                     | numeric (target for RQ1)                      |
| `threshold_reached`              | **0/1 integer** (target for RQ2)              |


---

## How to run

```bash
# macOS / Linux
python crossValidation_engine.py

# Windows (PowerShell)
python .\crossValidation_engine.py
```

**What you’ll see:**
For each task/model pair the script prints mean ± std of:

* **RQ1:** `R^2`, `MAE`
* **RQ2:** `balanced_accuracy`, `F1`

It also prints the class counts for the RQ2 label.

---

## What the script does

1. **Load & cast** `DataSet_complete.csv`
2. **Split features** into numeric and categorical.
3. **Preprocess** via a `ColumnTransformer`:

   * numeric: `SimpleImputer(median)` -> `StandardScaler`
   * categorical: `SimpleImputer(most_frequent)` -> `OneHotEncoder(drop="first")`
4. **Attribute reduction:** `VarianceThreshold(0.0)` removes constant columns.
5. **Cross-validation:**

   * RQ1: `KFold(n_splits=min(5, n))`
   * RQ2: `StratifiedKFold(n_splits=min(5, n))`
6. **Report** metrics to stdout.

---

## Customizing

* Change feature set: edit `feature_cols`, `numeric_features`, `categorical_features`.
* Add/remove models: edit `RQ1_models` / `RQ2_models`.
* Tweak CV or metrics: adjust the `evaluate_both_models` function.
* If your CSV path/name differs, modify the `pd.read_csv("DataSet_complete.csv")` line.
