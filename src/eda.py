import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/creditcard.csv")

print("Dataset Shape:", df.shape)
print("\nMissing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Fraud vs Normal
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.savefig("artifacts/fraud_distribution.png")
plt.close()

# Transaction Amount
plt.figure(figsize=(8, 5))
sns.boxplot(x="Class", y="Amount", data=df)
plt.title("Transaction Amount by Class")
plt.savefig("artifacts/amount_distribution.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("artifacts/correlation_heatmap.png")
plt.close()

print("\nEDA completed successfully!")
print("Charts saved in artifacts folder.")