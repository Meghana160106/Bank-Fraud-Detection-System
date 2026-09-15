import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


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

model = joblib.load("models/xgboost_optuna.pkl")

probabilities = model.predict_proba(X_test)[:, 1]

print("\nTesting different thresholds...")

results = []

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30,
                  0.35, 0.40, 0.45, 0.50, 0.55,
                  0.60, 0.65, 0.70]:

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    # False negatives are given a higher cost
    cost = (fp * 1) + (fn * 5)

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp,
        "false_negatives": fn,
        "cost": cost
    })


results_df = pd.DataFrame(results)

best = results_df.loc[
    results_df["cost"].idxmin()
]

print("\n===== THRESHOLD RESULTS =====")

print(results_df.to_string(index=False))

print("\n===== BEST THRESHOLD =====")

print("Threshold:", best["threshold"])
print("Precision:", round(best["precision"], 4))
print("Recall:", round(best["recall"], 4))
print("F1:", round(best["f1"], 4))
print("False Positives:", int(best["false_positives"]))
print("False Negatives:", int(best["false_negatives"]))
print("Cost:", int(best["cost"]))


with open("artifacts/best_threshold.txt", "w") as file:
    file.write(str(best["threshold"]))

results_df.to_csv(
    "artifacts/threshold_results.csv",
    index=False
)

print("\nThreshold saved successfully!")
print("File: artifacts/best_threshold.txt")