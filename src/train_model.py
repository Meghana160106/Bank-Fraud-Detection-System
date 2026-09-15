import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

print("Loading dataset...")

df = pd.read_csv("data/creditcard.csv")

print("Dataset shape:", df.shape)
print("Fraud cases:", df["Class"].sum())

X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE
print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("Training data after SMOTE:", X_train.shape)

# -------------------------------
# MODEL 1: LOGISTIC REGRESSION
# -------------------------------

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

print("\n===== LOGISTIC REGRESSION =====")
print(classification_report(y_test, lr_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr_prob))

# -------------------------------
# MODEL 2: RANDOM FOREST
# -------------------------------

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\n===== RANDOM FOREST =====")
print(classification_report(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Save Random Forest as main model
joblib.dump(rf_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n================================")
print("BEST MODEL: RANDOM FOREST")
print("Model saved successfully!")
print("================================")