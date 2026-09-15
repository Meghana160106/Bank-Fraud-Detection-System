import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="💳",
    layout="wide"
)

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("💳 Bank Fraud Detection System")
st.write("Machine Learning based detection of suspicious banking transactions")

st.divider()

st.subheader("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(f"File uploaded successfully — {len(df)} transactions found")

    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df.copy()

    try:
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]

        result = df.copy()

        result["Fraud Probability"] = probabilities
        result["Prediction"] = predictions

        result["Risk Level"] = result["Fraud Probability"].apply(
            lambda x: "🔴 HIGH" if x >= 0.7
            else "🟠 MEDIUM" if x >= 0.3
            else "🟢 LOW"
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        fraud_count = int((predictions == 1).sum())
        normal_count = int((predictions == 0).sum())

        col1.metric("Total Transactions", len(result))
        col2.metric("Normal Transactions", normal_count)
        col3.metric("Fraudulent Transactions", fraud_count)

        st.subheader("🚨 Transaction Analysis")

        st.dataframe(
            result,
            use_container_width=True
        )

        st.subheader("📊 Fraud Probability Distribution")

        st.bar_chart(
            result["Fraud Probability"].head(50)
        )

    except Exception as e:
        st.error("Invalid transaction format.")
        st.write("Expected columns:")
        st.write(list(X.columns))

else:
    st.info("Upload a transaction CSV file to start fraud detection.")