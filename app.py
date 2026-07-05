from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("models/employee_attrition_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    # First time page loads
    if request.method == "GET":
        return render_template("index.html")

    # Get form data
    age = int(request.form["Age"])
    business = request.form["BusinessTravel"]
    department = request.form["Department"]
    distance = int(request.form["DistanceFromHome"])
    education = int(request.form["Education"])
    environment = int(request.form["EnvironmentSatisfaction"])
    gender = request.form["Gender"]
    jobrole = request.form["JobRole"]
    jobsatisfaction = int(request.form["JobSatisfaction"])
    marital = request.form["MaritalStatus"]
    income = int(request.form["MonthlyIncome"])
    overtime = request.form["OverTime"]

    # Create dataframe
    input_df = pd.DataFrame([{
        "Age": age,
        "BusinessTravel": business,
        "Department": department,
        "DistanceFromHome": distance,
        "Education": education,
        "EnvironmentSatisfaction": environment,
        "Gender": gender,
        "JobRole": jobrole,
        "JobSatisfaction": jobsatisfaction,
        "MaritalStatus": marital,
        "MonthlyIncome": income,
        "OverTime": overtime
    }])

    # Prediction
    prediction = model.predict(input_df)
    confidence = model.predict_proba(input_df).max() * 100

    # Insights
    insights = []

    if overtime == "Yes":
        insights.append("Employee frequently works overtime, which may contribute to burnout.")

    if income < 4000:
        insights.append("Monthly income is relatively low and may affect retention.")

    if distance > 20:
        insights.append("Long commuting distance may impact work-life balance.")

    if jobsatisfaction <= 2:
        insights.append("Job satisfaction is below average.")

    if environment <= 2:
        insights.append("Work environment satisfaction is low.")

    # Prediction Result
    if prediction[0] == "Yes":

        result = " Employee is likely to Leave"
        risk = "High"

        recommendations = [
            "Schedule a one-on-one discussion with the employee.",
            "Review workload and reduce overtime if possible.",
            "Discuss career growth and promotion opportunities.",
            "Consider recognition or compensation improvements."
        ]

    else:

        result = "Employee is likely to Stay"
        risk = "Low"

        recommendations = [
            "Continue recognizing good performance.",
            "Provide learning and development opportunities.",
            "Maintain a healthy work-life balance.",
            "Conduct regular feedback sessions."
        ]

    return render_template(
        "index.html",
        prediction=result,
        probability=round(confidence, 2),
        risk=risk,
        insights=insights,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)