import pandas as pd
import os

reference = pd.read_csv("data/creditcard.csv")

current = reference.sample(
    5000,
    random_state=10
)

reference_mean = reference["Amount"].mean()
current_mean = current["Amount"].mean()

difference = abs(
    current_mean - reference_mean
) / reference_mean

if difference > 0.20:
    status = "DRIFT DETECTED"
else:
    status = "NO SIGNIFICANT DRIFT"

print("Reference Amount Mean:", reference_mean)
print("Current Amount Mean:", current_mean)
print(status)
print("Drift monitoring completed!")

# Save result for dashboard
os.makedirs("artifacts", exist_ok=True)

with open("artifacts/drift_status.txt", "w") as file:
    file.write(status)