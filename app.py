import pandas as pd
import pickle
from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
model = pickle.load(open("best_model.pkl", "rb"))

# All training features from model
FEATURES = list(model.feature_names_in_)

# Airline columns automatically picked
AIRLINE_COLS = [c for c in FEATURES if c.startswith("Airline_")]

# Dropdown airlines → exact training airline names
DISPLAY_AIRLINES = [
    "IndiGo", "Air India", "Vistara",
    "SpiceJet", "GoAir", "Trujet"
]

DISPLAY_TO_MODEL = {
    "IndiGo": "Airline_IndiGo",
    "Air India": "Airline_Air India",
    "Vistara": "Airline_Vistara",
    "SpiceJet": "Airline_SpiceJet",
    "GoAir": "Airline_GoAir",
    "Trujet": "Airline_Trujet"
}

# City Encoding
source_encode = {"DEL":0, "BOM":3, "BLR":2, "CCU":1, "HYD":4, "COK":5}
dest_encode   = {"DEL":0, "BOM":1, "BLR":2, "CCU":3, "HYD":4, "COK":5}

def get_route_code(s, d):
    return s * 10 + d


# -----------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html")


# -----------------------------------
@app.route("/predict-api", methods=["POST"])
def predict_api():

    try:
        data = request.get_json()

        src     = data["source_code"]
        dest    = data["dest_code"]
        day     = int(data["day"])
        month   = int(data["month"])
        year    = int(data["year"])
        dep_hr  = int(data["dep_hour"])
        arr_hr  = int(data["arr_hour"])

        s = source_encode[src]
        d = dest_encode[dest]

        duration_min = (arr_hr - dep_hr) * 60
        if duration_min < 0:
            duration_min += 1440

        route = get_route_code(s, d)

        results = []

        for airline in DISPLAY_AIRLINES:

            # ----------------------------------------
            # CREATE EMPTY ROW WITH ALL ZERO FEATURES
            # ----------------------------------------
            row = {col: 0 for col in FEATURES}

            # Fill numeric features
            row["Source"]            = s
            row["Destination"]       = d
            row["Route"]             = route
            row["Total_Stops"]       = 0
            row["Journey_Day"]       = day
            row["Journey_Month"]     = month
            row["Dep_Hour"]          = dep_hr
            row["Dep_Min"]           = 0
            row["Arr_Hour"]          = arr_hr
            row["Arr_Min"]           = 0
            row["Duration_Minutes"]  = duration_min

            # Airline one-hot
            col_name = DISPLAY_TO_MODEL[airline]
            if col_name in row:
                row[col_name] = 1

            df_row = pd.DataFrame([row])

            price = float(model.predict(df_row)[0])

            results.append({
                "airline_name": airline,
                "duration": duration_min,
                "stops": 0,
                "price": f"{round(price):,}"
            })

        # Sort cheapest
        results = sorted(results, key=lambda x: int(x["price"].replace(",", "")))

        date_str = pd.to_datetime(f"{year}-{month}-{day}").strftime("%d %b %Y")

        return jsonify({
            "route": f"{src} → {dest}",
            "journey_date": date_str,
            "results": results,
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"})


# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
