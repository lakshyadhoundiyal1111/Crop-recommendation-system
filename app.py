import joblib
import numpy as np
import finder
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load the model
model = joblib.load('smart-agriculture.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predictor')
def predictor():
    return render_template('predictor.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract from the POST request
        data = request.json
        features = [data['N'], data['K'], data['P'], data['temperature'],
                    data['humidity'], data['ph'], data['rainfall']]

        # Convert features to numpy array and reshape for prediction
        features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]
        return jsonify({'crop': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route("/service2")
def helper():
    return render_template('help.html')

@app.route("/find-crop", methods=["POST"])
def find_crop():
    # Parse JSON input from the request
    data = request.get_json()  # Expecting JSON payload from the client
    crop_name = data.get("input")  # Extract the "input" field from the JSON

    if not crop_name:
        return {"error": "No crop name provided."}, 400  # Return JSON error

    # Call the find_crop function to get the result
    result = finder.find_crop(crop_name)

    # Return the result as a JSON response
    return {"crop_name": crop_name, "result": result}

if __name__ == '__main__':
    app.run(debug=True)
