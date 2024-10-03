from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import time
import requests
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Load the dataset and pre-trained model
data = pd.read_csv('FINAL.csv')
geo_data = pd.read_csv('location.csv')
data = pd.merge(data, geo_data, how='left', on='District')

# Label encoders for categorical features
le_district = LabelEncoder()
le_commodity = LabelEncoder()
le_variety = LabelEncoder()

data['District_Encoded'] = le_district.fit_transform(data['District'])
data['Commodity_Encoded'] = le_commodity.fit_transform(data['Commodity'])
data['Variety_Encoded'] = le_variety.fit_transform(data['Variety'])

# Mapping months to numbers
month_name_to_number = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
    'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
    'November': 11, 'December': 12
}
data['Month'] = data['Month_Name'].map(month_name_to_number)
data = data.dropna(subset=['Commodity_Encoded', 'Variety_Encoded', 'District_Encoded', 'Year', 'Month', 'Modal_Price'])

# Feature selection
features = ['Commodity_Encoded', 'Variety_Encoded', 'District_Encoded', 'Year', 'Month']
target = 'Modal_Price'

X = data[features]
y = data[target]

# Split data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load or train the model
try:
    with open("rf_regressor.H5", "rb") as file:
        rf_model = pickle.load(file)
except FileNotFoundError:
    # Train a Random Forest model if not found
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    with open("rf_regressor.H5", "wb") as file:
        pickle.dump(rf_model, file)

# Function to fetch latitude and longitude using OpenStreetMap Nominatim API
def get_lat_lon_from_district(district_name):
    url = f"https://nominatim.openstreetmap.org/search?q={district_name}&format=json"
    response = requests.get(url)
    data = response.json()
    if data:
        lat = float(data[0]['lat'])
        lon = float(data[0]['lon'])
        return lat, lon
    else:
        raise ValueError(f"Could not find coordinates for {district_name}")

# Updated haversine function using geopy
def calculate_distance(lat1, lon1, lat2, lon2):
    loc1 = (lat1, lon1)
    loc2 = (lat2, lon2)
    return geodesic(loc1, loc2).kilometers

# Prediction function incorporating dynamic latitude and longitude fetching
def predict_prices(chosen_commodity, chosen_variety, chosen_district, quantity, chosen_month, chosen_year):
    try:
        current_lat, current_lon = get_lat_lon_from_district(chosen_district)
    except ValueError as e:
        return {"error": str(e)}

    district_encoded = le_district.transform([chosen_district])[0]
    commodity_encoded = le_commodity.transform([chosen_commodity])[0]
    variety_encoded = le_variety.transform([chosen_variety])[0]

    input_data = pd.DataFrame({
        'Commodity_Encoded': [commodity_encoded],
        'Variety_Encoded': [variety_encoded],
        'District_Encoded': [district_encoded],
        'Year': [chosen_year],
        'Month': [chosen_month]
    })

    predicted_price = rf_model.predict(input_data)[0]
    total_cost = (predicted_price / 100) * quantity

    # Calculate distances from nearby districts
    data['distance'] = data.apply(lambda row: calculate_distance(current_lat, current_lon, row['Latitude'], row['Longitude']), axis=1)

    # Get nearby districts with best prices
    nearby_districts = data[(data['Commodity'] == chosen_commodity) &
                            (data['Variety'] == chosen_variety) &
                            (data['District'] != chosen_district)] \
                       .sort_values(by=['distance', 'Modal_Price']).drop_duplicates(subset='District').head(3)

    nearby_districts_info = nearby_districts[['District', 'distance', 'Modal_Price']].to_dict(orient='records')

    return {
        "predictedPrice": predicted_price,
        "totalCost": total_cost,
        "nearbyDistricts": nearby_districts_info
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    district = data['district']
    commodity = data['commodity']
    variety = data['variety']
    amount = float(data['amount'])
    month = int(data['month'])
    year = int(data['year'])

    # Call the prediction logic
    result = predict_prices(commodity, variety, district, amount, month, year)

    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
