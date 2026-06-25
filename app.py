from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("flood_model.pkl", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    temp = float(request.form["Temp"])
    humidity = float(request.form["Humidity"])
    cloud = float(request.form["Cloud"])
    annual = float(request.form["Annual"])
    avgjune = float(request.form["AvgJune"])
    sub = float(request.form["Sub"])

    features = np.array([[temp,
                          humidity,
                          cloud,
                          annual,
                          avgjune,
                          sub]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "High Flood Risk"
    else:
        result = "No Flood Risk"

    return render_template("result.html",
                           prediction=result)


if __name__ == "__main__":
    app.run(debug=True)