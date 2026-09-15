FROM python:3.11

WORKDIR /app

COPY . .

RUN pip install pandas numpy scikit-learn xgboost joblib fastapi uvicorn streamlit

EXPOSE 8000
EXPOSE 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


