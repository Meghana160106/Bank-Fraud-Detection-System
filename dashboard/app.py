import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =========================
# LOAD MODELS
# =========================

model = joblib.load("models/xgboost_optuna.pkl")
rf_model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Best threshold
try:
    with open("artifacts/best_threshold.txt", "r") as file:
        BEST_THRESHOLD = float(file.read())
except:
    BEST_THRESHOLD = 0.5


# =========================
# TITLE
# =========================

st.title("💳 Bank Fraud Detection System")
st.write(
    "Machine Learning based detection and analysis of suspicious banking transactions"
)

st.divider()


# =========================
# MODEL PERFORMANCE
# =========================

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

col1.metric(
    "XGBoost ROC-AUC",
    "0.9793"
)

col2.metric(
    "Random Forest ROC-AUC",
    "0.9734"
)

st.divider()


# =========================
# UPLOAD DATA
# =========================

st.subheader("📂 Upload Transaction Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(
        f"File uploaded successfully — {len(df)} transactions found"
    )

    # Remove target column if present
    if "Class" in df.columns:
        X = df.drop("Class", axis=1)
    else:
        X = df.copy()

    try:

        # =========================
        # XGBOOST PREDICTION
        # =========================

        probabilities = model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= BEST_THRESHOLD
        ).astype(int)

        result = df.copy()

        result["Fraud Probability"] = probabilities

        result["Prediction"] = predictions

        # =========================
        # RISK LEVEL
        # =========================

        result["Risk Level"] = result["Fraud Probability"].apply(
            lambda x:
            "🔴 HIGH" if x >= 0.70
            else "🟠 MEDIUM" if x >= 0.30
            else "🟢 LOW"
        )

        # =========================
        # SUMMARY
        # =========================

        total_transactions = len(result)

        fraud_count = int(
            (predictions == 1).sum()
        )

        normal_count = int(
            (predictions == 0).sum()
        )

        high_risk = int(
            (result["Risk Level"] == "🔴 HIGH").sum()
        )

        medium_risk = int(
            (result["Risk Level"] == "🟠 MEDIUM").sum()
        )

        low_risk = int(
            (result["Risk Level"] == "🟢 LOW").sum()
        )


        # =========================
        # DASHBOARD METRICS
        # =========================

        st.subheader("📌 Transaction Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            total_transactions
        )

        col2.metric(
            "Normal Transactions",
            normal_count
        )

        col3.metric(
            "Fraudulent Transactions",
            fraud_count
        )

        col4.metric(
            "Detection Threshold",
            f"{BEST_THRESHOLD:.2f}"
        )


        # =========================
        # RISK SUMMARY
        # =========================

        st.subheader("🚨 Risk Distribution")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🔴 HIGH Risk",
            high_risk
        )

        col2.metric(
            "🟠 MEDIUM Risk",
            medium_risk
        )

        col3.metric(
            "🟢 LOW Risk",
            low_risk
        )


        # =========================
        # FRAUD PROBABILITY
        # =========================

        st.divider()

        st.subheader("📈 Fraud Probability Chart")

        chart_data = result[
            ["Fraud Probability"]
        ].head(100)

        st.line_chart(
            chart_data
        )


        # =========================
        # MODEL COMPARISON
        # =========================

        st.subheader("🤖 Model Comparison")

        comparison = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "Random Forest",
                "XGBoost"
            ],
            "ROC-AUC": [
                0.9730,
                0.9734,
                0.9793
            ]
        })

        st.bar_chart(
            comparison.set_index("Model")
        )

        st.dataframe(
            comparison,
            use_container_width=True
        )


        # =========================
        # ANOMALY DETECTION
        # =========================

        st.subheader("🔎 Anomaly Detection")

        if os.path.exists(
            "models/isolation_forest.pkl"
        ):

            isolation_model = joblib.load(
                "models/isolation_forest.pkl"
            )

            anomaly_predictions = isolation_model.predict(X)

            anomalies = (
                anomaly_predictions == -1
            ).sum()

            normal_points = (
                anomaly_predictions == 1
            ).sum()

            col1, col2 = st.columns(2)

            col1.metric(
                "Detected Anomalies",
                int(anomalies)
            )

            col2.metric(
                "Normal Behaviour",
                int(normal_points)
            )

        else:

            st.info(
                "Isolation Forest model will appear here after Week 4 anomaly detection is completed."
            )


        # =========================
        # SHAP EXPLANATION
        # =========================

        st.subheader("🧠 SHAP Explanation")

        if os.path.exists(
            "artifacts/shap_summary.png"
        ):

            st.image(
                "artifacts/shap_summary.png",
                caption="XGBoost Feature Importance using SHAP"
            )

        else:

            st.info(
                "SHAP explanation will appear here."
            )


        # =========================
        # DRIFT MONITORING
        # =========================

        st.subheader("📡 Data Drift Monitoring")

        if os.path.exists(
            "artifacts/drift_status.txt"
        ):

            with open(
                "artifacts/drift_status.txt",
                "r"
            ) as file:

                drift_status = file.read()

            st.success(
                drift_status
            )

        else:

            st.info(
                "Drift monitoring will appear here after the drift detection module is completed."
            )


        # =========================
        # TRANSACTION TABLE
        # =========================

        st.divider()

        st.subheader("🚨 Transaction Analysis")

        st.dataframe(
            result,
            use_container_width=True
        )


        # =========================
        # DOWNLOAD RESULTS
        # =========================

        csv = result.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Analysis Results",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv"
        )


    except Exception as e:

        st.error(
            "Invalid transaction format."
        )

        st.write(
            "Error:",
            str(e)
        )

else:

    st.info(
        "Upload a transaction CSV file to start fraud detection."
    )