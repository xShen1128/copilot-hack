# Create a base Flask server

import pickle
from pathlib import Path

from flask import Flask, request, jsonify

app = Flask(__name__)

# Resolve paths against this file's directory instead of the current working
# directory, so the server can be started from anywhere.
BASE_DIR = Path(__file__).resolve().parent

# Enable cors
@app.after_request
def after_request(response):
    """
    Enable CORS
    """
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    return response


def load_airports():
    """
    Read airports.csv from the server directory and return a list of
    {'id': int, 'name': str} dictionaries, sorted by name.
    """
    with (BASE_DIR / "airports.csv").open("r", encoding="utf-8") as file:
        rows = file.readlines()

    # Remove first line of airports (the header row)
    rows.pop(0)

    # Create list with dictionary of airports
    # First value is id, second is name
    airports = [
        {'id': int(row.split(',')[0]), 'name': row.split(',')[1].strip()}
        for row in rows
        if row.strip()
    ]

    # Sort by name
    return sorted(airports, key=lambda k: k['name'])


# Load model from pickle file
with (BASE_DIR / "model.pkl").open("rb") as model_file:
    model = pickle.load(model_file)

# Model takes two parameters - day of week and airport id, then returns a prediction of flight delay
@app.route('/predict', methods=['GET'])
def predict():
    """
    Takes two parameters - day of week and airport id, then returns a prediction of flight delay
    """
    day_of_week_param = request.args.get('day_of_week')
    airport_id_param = request.args.get('airport_id')

    # Both parameters are required and must be integers
    if day_of_week_param is None or airport_id_param is None:
        return jsonify({"error": "day_of_week and airport_id must be integers"}), 400

    try:
        day_of_week = int(day_of_week_param)
        airport_id = int(airport_id_param)
    except (TypeError, ValueError):
        return jsonify({"error": "day_of_week and airport_id must be integers"}), 400

    # Day of week must be between 1 and 7
    if day_of_week < 1 or day_of_week > 7:
        return jsonify({"error": "day_of_week must be an integer between 1 and 7"}), 400

    # Airport id must exist in airports.csv
    known_airport_ids = {airport['id'] for airport in load_airports()}
    if airport_id not in known_airport_ids:
        return jsonify({"error": f"airport_id {airport_id} was not found in airports.csv"}), 400

    prediction = model.predict_proba([[day_of_week, airport_id]])[0]
    
    # Split prediction string by space
    prediction = str(prediction).split(' ')

    # store first value from prediction as certainty, and remove the first character
    certainty = float(prediction[0][2:])

    # store second value from prediction as delay, and remove the last character
    delay = float(prediction[1][:-1])

    # return prediction as json
    return jsonify({'certainty': certainty, 'delay': delay})

# Create a new route called airports with method of get
@app.route('/airports', methods=['GET'])
def airports():
    return jsonify(load_airports())

if __name__ == '__main__':
    app.run(debug=True)