
# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Step 2: Load dataset
data = pd.read_csv("churn.csv")

# 🔴 IMPORTANT: Drop CustomerID (never use IDs in ML)
if "CustomerID" in data.columns:
    data = data.drop(columns=["CustomerID"])

# Step 3: Separate target
y = data["Churn"]

# Convert Yes/No → 1/0 (if needed)
if y.dtype == object:
    y = y.map({"Yes": 1, "No": 0})

# Step 4: Features
X = data.drop("Churn", axis=1)

# Convert categorical columns to numbers
X = pd.get_dummies(X, drop_first=True)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 7: Evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("✅ Model Accuracy:", accuracy)

# Step 8: Save model + feature columns

joblib.dump(
    {
        "model": model,
        "columns": X.columns.tolist(),
        "feature_means": X.mean().to_dict()
    },
    "model.pkl"
)

print("✅ Model, columns, and feature means saved")
