import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd


app = Flask(__name__)

model = joblib.load('trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    data = request.form
    
    City = str(data['City'])
    type1 = str(data['type'])
    condition = str(data['condition'])
    entranceDate=str(data['entranceDate'])
    room_number = float(data['room_number'])
    Area = float(data['Area'])
    floor = float(data['floor'])
    hasElevator = int(data.get('hasElevator', 0))
    hasParking = int(data.get('hasParking', 0))
    hasBars = int(data.get('hasBars', 0))
    hasStorage = int(data.get('hasStorage', 0))
    hasAirCondition = int(data.get('hasAirCondition', 0))
    hasBalcony = int(data.get('hasBalcony', 0))
    hasMamad = int(data.get('hasMamad', 0))
    handicapFriendly = int(data.get('handicapFriendly', 0))

    
    # Create a feature DataFrame
    data = {'City': [City], 'type': [type1], 'condition': [condition],'entranceDate':[entranceDate],
            'room_number': [room_number], 'Area': [Area], 'hasElevator': [hasElevator],
            'hasParking': [hasParking], 'hasBars': [hasBars], 'hasStorage': [hasStorage],
            'hasAirCondition': [hasAirCondition], 'hasBalcony': [hasBalcony], 'hasMamad': [hasMamad],
            'handicapFriendly': [handicapFriendly], 'floor': [floor]}
    
    df = pd.DataFrame(data)
    
    # Make a prediction
    y_pred = round(model.predict(df)[0])
    
    # Return the predicted price
    return render_template('index.html', price=y_pred)
@app.errorhandler(500)
def handle_internal_server_error(error):
    return f"Internal Server Error: {error}", 500
if __name__ == '__main__':
    app.run()