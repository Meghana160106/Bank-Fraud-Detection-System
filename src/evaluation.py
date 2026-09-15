import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from imblearn.over_sampling import SMOTE

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

scaler = joblib.load("models/scaler.pkl")
model = joblib.load("models/fraud_model.pkl")

X_test_scaled = scaler.transform(X_test)

predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)[:, 1]

print("\n===== MODEL EVALUATION =====")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("ROC-AUC:",
      round(roc_auc_score(y_test, probabilities), 4))

print("PR-AUC:",
      round(average_precision_score(y_test, probabilities), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions
).ravel()

fpr = fp / (fp + tn)

print("\nFalse Positive Rate:",
      round(fpr * 100, 4), "%")

print("\nEvaluation completed successfully!")