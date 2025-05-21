from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def index():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[col]) for col in [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
            'ca', 'thal'
        ]]
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        result = "At Risk of Heart Disease" if prediction == 1 else "Not at Risk"
        return render_template("form.html", prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
