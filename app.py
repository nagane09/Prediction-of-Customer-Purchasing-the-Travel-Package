from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)


MODEL_PATH = "model.joblib"
PREPROCESSOR_PATH = "preprocessor.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError("Model or preprocessor files are missing!")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = ""
    
    if request.method == "POST":
        try:
            input_data = pd.DataFrame([{
                "Age": int(request.form["Age"]),
                "Gender": request.form["Gender"],
                "MaritalStatus": request.form["MaritalStatus"],
                "ProductPitched": request.form["ProductPitched"],
                "PreferredPropertyStar": int(request.form["PreferredPropertyStar"]),
                "NumberOfTrips": int(request.form["NumberOfTrips"]),
                "TypeofContact": request.form["TypeofContact"],
                "DurationOfPitch": int(request.form["DurationOfPitch"]),
                "NumberOfFollowups": int(request.form["NumberOfFollowups"]),
                "PitchSatisfactionScore": int(request.form["PitchSatisfactionScore"]),
                "Designation": request.form["Designation"],
                "CityTier": int(request.form["CityTier"]),
                "Occupation": request.form["Occupation"],
                "Passport": int(request.form["Passport"]),
                "OwnCar": int(request.form["OwnCar"]),
                "MonthlyIncome": float(request.form["MonthlyIncome"]),
                "TotalVisiting": int(request.form["TotalVisiting"])
            }])

            transformed = preprocessor.transform(input_data)
            prediction = model.predict(transformed)[0]

            if prediction == 1:
                prediction_text = "✔ Customer is LIKELY to purchase the travel package."
            else:
                prediction_text = "✘ Customer is NOT likely to purchase the travel package."

        except Exception as e:
            prediction_text = f"Error: {e}"

    return render_template("index.html", prediction_text=prediction_text)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=True)
