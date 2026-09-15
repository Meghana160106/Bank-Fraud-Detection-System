def calculate_risk(fraud_probability, anomaly_score):

    score = (
        fraud_probability * 70
        + anomaly_score * 30
    )

    if score >= 70:
        level = "HIGH"
    elif score >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    return round(score, 2), level


if __name__ == "__main__":

    score, level = calculate_risk(
        fraud_probability=0.85,
        anomaly_score=0.60
    )

    print("Risk Score:", score)
    print("Risk Level:", level)