import pandas as pd
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


print("Loading dataset...")

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Dataset loaded.")
print("Starting Optuna tuning...")


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", 100, 300
        ),
        "max_depth": trial.suggest_int(
            "max_depth", 3, 8
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2
        ),
        "subsample": trial.suggest_float(
            "subsample", 0.7, 1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.7, 1.0
        ),
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    return roc_auc_score(y_test, probabilities)


study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=20
)

print("\n===== OPTUNA RESULTS =====")

print("Best ROC-AUC:")
print(round(study.best_value, 4))

print("\nBest Parameters:")

for key, value in study.best_params.items():
    print(f"{key}: {value}")


print("\nTraining optimized XGBoost...")

best_params = study.best_params

best_model = XGBClassifier(
    **best_params,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)

joblib.dump(
    best_model,
    "models/xgboost_optuna.pkl"
)

print("\nOptimized model saved successfully!")
print("File: models/xgboost_optuna.pkl")