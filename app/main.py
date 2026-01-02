
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# =========================
# Load model artifacts
# =========================
data = joblib.load("model.pkl")
model = data["model"]
columns = data["columns"]
feature_means = data["feature_means"]


# =========================
# Input Schema
# =========================
class CustomerInput(BaseModel):
    CustomerID: int
    Age: int
    Gender: str
    Tenure: int
    Usage_Frequency: int
    Support_Calls: int
    Payment_Delay: int
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: int


# =========================
# Drift Detection Function
# =========================
def detect_drift(input_df, means, threshold=0.5):
    drifted_features = []

    for col in input_df.columns:
        if col in means:
            mean_val = means[col]
            input_val = input_df[col].iloc[0]

            if mean_val != 0:
                diff_ratio = abs(input_val - mean_val) / abs(mean_val)
                if diff_ratio > threshold:
                    drifted_features.append(col)

    return drifted_features


# =========================
# Routes
# =========================
@app.get("/")
def home():
    return {"message": "ML Model API is running"}


@app.post("/predict")
def predict(data: CustomerInput):
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Remove ID column (not used in training)
    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])

    # Apply same preprocessing as training
    df = pd.get_dummies(df)

    # Add missing columns
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns
    df = df[columns]

    # Prediction
    probability = model.predict_proba(df)[0][1]
    prediction = int(probability > 0.5)

    # Drift detection
    drifted_features = detect_drift(df, feature_means, threshold=10)  # instead of 0.5


    # Logging (monitoring)
    with open("predictions.log", "a") as f:
        f.write(
            f"input={df.to_dict()} | "
            f"prob={round(probability,3)} | "
            f"prediction={prediction} | "
            f"drift={drifted_features}\n"
        )

    return {
        "Churn_Prediction": prediction,
        "Churn_Probability": round(probability, 3),
        "Drift_Detected": len(drifted_features) > 0,
        "Drifted_Features": drifted_features,
        "Meaning": "Customer will churn" if prediction == 1 else "Customer will NOT churn"
    }


@app.get("/stats")
def stats():
    try:
        with open("predictions.log", "r") as f:
            lines = f.readlines()

        return {
            "total_predictions": len(lines),
            "last_5_predictions": lines[-5:]
        }
    except FileNotFoundError:
        return {"message": "No predictions logged yet"}
