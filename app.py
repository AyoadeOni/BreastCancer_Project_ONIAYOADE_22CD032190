from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("model/breast_cancer_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            features = [
                float(request.form["radius_mean"]),
                float(request.form["texture_mean"]),
                float(request.form["perimeter_mean"]),
                float(request.form["area_mean"]),
                float(request.form["compactness_mean"])
            ]

            input_data = np.array([features])
            input_scaled = scaler.transform(input_data)

            result = model.predict(input_scaled)[0]

            prediction = "Malignant" if result == 1 else "Benign"

        except:
            prediction = "Invalid input"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
