import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

print("Loading data and XGBoost model...")

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)

model = joblib.load("models/xgboost_model.pkl")

# Use a small sample for faster SHAP analysis
X_sample = X.sample(1000, random_state=42)

print("Calculating SHAP values...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.tight_layout()

plt.savefig(
    "artifacts/shap_summary.png",
    bbox_inches="tight"
)

plt.close()

print("\nSHAP analysis completed successfully!")
print("Chart saved: artifacts/shap_summary.png")