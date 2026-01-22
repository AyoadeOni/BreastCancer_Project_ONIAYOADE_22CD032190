import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------------------
# 1. Load Dataset
# ----------------------------------
data = pd.read_csv("data.csv")  # Kaggle file

# ----------------------------------
# 2. Feature Selection
# ----------------------------------
features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "compactness_mean"
]

X = data[features]
y = data["diagnosis"]

# ----------------------------------
# 3. Encode Target Variable
# B -> 0 (Benign)
# M -> 1 (Malignant)
# ----------------------------------
y = y.map({"B": 0, "M": 1})

# ----------------------------------
# 4. Handle Missing Values
# ----------------------------------
X = X.fillna(X.mean())

# ----------------------------------
# 5. Train-Test Split
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------
# 6. Feature Scaling (MANDATORY)
# ----------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------
# 7. Train Model (Logistic Regression)
# ----------------------------------
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# ----------------------------------
# 8. Model Evaluation
# ----------------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")

# ----------------------------------
# 9. Save Model and Scaler
# ----------------------------------
joblib.dump(model, "model/breast_cancer_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("Model and scaler saved successfully.")
