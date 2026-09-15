import pandas as pd
import matplotlib.pyplot as plt

models = {
    "Logistic Regression": 0.9730,
    "Random Forest": 0.9734,
    "XGBoost": 0.9793
}

names = list(models.keys())
scores = list(models.values())

plt.figure(figsize=(8, 5))

bars = plt.bar(names, scores)

plt.title("Model ROC-AUC Comparison")
plt.ylabel("ROC-AUC Score")
plt.ylim(0.95, 1.00)

for bar, score in zip(bars, scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        score,
        f"{score:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("artifacts/model_comparison.png")
plt.close()

print("Model comparison chart created successfully!")
print("\nBest Model: XGBoost")
print("ROC-AUC: 0.9793")