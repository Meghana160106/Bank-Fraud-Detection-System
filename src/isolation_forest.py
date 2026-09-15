import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest

df = pd.read_csv("data/creditcard.csv")

X = df.drop("Class", axis=1)

model = IsolationForest(
    n_estimators=100,
    contamination=0.002,
    random_state=42
)

model.fit(X)

joblib.dump(model, "models/isolation_forest.pkl")

print("Isolation Forest saved successfully!")