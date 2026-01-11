import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Complete_configs.csv")

# feature columns and targets
feature_cols = [
    "env_name",
    "learning_rate",
    "batch_size",
    "nn_arch_depth",
    "cpu_cores_logical",
    "ram_total_gb",
]

target_RQ1 = "final_perf"            
target_RQ2  = "steps_to_threshold"    

#PREPROCESSING
#1.Type casting: 
numeric_casted= ["learning_rate", "batch_size","nn_arch_depth","cpu_cores_logical","ram_total_gb", target_RQ1, target_RQ2]
for c in numeric_casted:
    if c in df.columns: 
        df[c]=pd.to_numeric(df[c],errors="coerce")

X = df[feature_cols].copy()
y_perf = df[target_RQ1].values
y_reached = (~df[target_RQ2].isna()).astype(int).values

# 2. numeric vs categorical features
numeric_features = [
    "learning_rate",
    "batch_size",
    "nn_arch_depth",
    "cpu_cores_logical",
    "ram_total_gb",
]

categorical_features = ["env_name"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            #3. Standardization
            ("scaler", StandardScaler()),
        ]), numeric_features),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            #4. Encoding categoricals
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
        ]), categorical_features),
    ]
)
# 5. Attribute reduction: removing strictly-constant features
feature_filter = VarianceThreshold(threshold=0.0)

# MODELING
# from  only the Linear regression of phase 2, we're now adding the Random Forest as planned
RQ1_models = {
    "LinearRegression": LinearRegression(),
    "RandomForestRegressor": RandomForestRegressor(random_state=0),
}
RQ2_models = {
    "LogisticRegression_balanced": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForestClassifier_balanced": RandomForestClassifier(random_state=0, class_weight="balanced"),
}

#EVALUATION
def evaluate_both_models(X, y, models, task_name, task_type):

    if task_type == "regression":
        cv = KFold(n_splits=min(5, len(X)), shuffle=True, random_state=0)
        scoring = {"r2": "r2", "mae": "neg_mean_absolute_error"}
    elif task_type == "classification":
        cv = StratifiedKFold(n_splits=min(5, len(X)), shuffle=True, random_state=0)
        scoring = {"bal_acc": "balanced_accuracy", "f1": "f1"}
    
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocess),
            ("filter", feature_filter),
            ("model", model),
        ])
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, error_score="raise")
        print(f"\n=== {task_name} ({task_type}) ===")
        print(f"{name}:")

        if task_type == "regression":
            r2 = scores["test_r2"]
            mae = -scores["test_mae"]  # negate because sklearn returns negative MAE
            print(f"  R^2 mean = {r2.mean():.3f}, std = {r2.std():.3f}")
            print(f"  MAE mean = {mae.mean():.3f}, std = {mae.std():.3f}")
        else:
            bal = scores["test_bal_acc"]
            f1 = scores["test_f1"]
            print(f"  Balanced Acc mean = {bal.mean():.3f}, std = {bal.std():.3f}")
            print(f"  F1 mean = {f1.mean():.3f}, std = {f1.std():.3f}")
#6. hadnling missing values for RQ1
mask = ~np.isnan(y_perf)
X_RQ1 = X.loc[mask]
y_RQ1 = y_perf[mask]

evaluate_both_models(X_RQ1, y_RQ1, RQ1_models, "final_perf (RQ1)", "regression")
evaluate_both_models(X, y_reached, RQ2_models, "is threshold reahced (RQ2)", "classification")

df_plot = df[["env_name", "final_perf"]]
df_plot.boxplot(column="final_perf", by="env_name", rot=45)
plt.xlabel("env_name")
plt.ylabel("final_perf")
plt.title("environment vs Final Performance")
plt.show()
print(np.bincount(y_reached))