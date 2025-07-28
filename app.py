from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

#Defining Locations
import json
with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

locations = data_columns[3:]  # everything after bhk, area, individual house

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bhk = int(request.form['bhk'])
        area = float(request.form['area'])
        ind_house = int(request.form['individual house'])
        location = request.form['location'].lower()

        x = np.zeros(len(data_columns))
        x[0] = bhk
        x[1] = area
        x[2] = ind_house

        if location in data_columns:
            loc_index = data_columns.index(location)
            x[loc_index] = 1

        prediction = model.predict([x])[0]
        adjusted = prediction * 1.6  # Using a 45% uplift
        prediction_lakhs = round(adjusted / 100000, 2)
        result = f"Estimated Price (adjusted): ₹ {prediction_lakhs} lakhs"

    except Exception as e:
            result = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=result, locations=locations)

if __name__ == "__main__":
    app.run(debug=True)