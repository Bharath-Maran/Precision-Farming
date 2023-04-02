# Importing essential libraries and modules

from flask import Flask, render_template, request, Markup
import numpy as np
import pandas as pd
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from sklearn.preprocessing import LabelEncoder

# ==============================================================================================
#------------------FITTING LABEL ENCODER-------------------------------------------------------
csv_file_path = 'dataset/crop_recommendation.csv'
df = pd.read_csv(csv_file_path)
prediction_labels = df['label']
le = LabelEncoder()
le.fit(prediction_labels)
# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading crop recommendation model

crop_recommendation_model_path = 'models/XGBoost.pkl' #New Version
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

# ===============================================================================================
# ------------------------------------ FLASK APP -------------------------------------------------


app = Flask(__name__)

# render home page
@ app.route('/')
def home():
    title = 'Precision Farming - Home'
    return render_template('index.html', title=title)

# render crop recommendation form page
@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Precision Farming - Crop Recommendation'
    return render_template('crop.html', title=title)

# render module page
@ app.route('/modules')
def modules():
    title = 'Precision Farming - Modules'
    return render_template('modules.html', title=title)

# render data collection page
@ app.route('/data-collection')
def data_collection():
    title = 'Precision Farming - Data Collection'
    return render_template('data-collection.html', title=title)

# render performance metrics page
@ app.route('/performance-metrics')
def performance_metrics():
    title = 'Precision Farming - Performance Metrics'
    return render_template('performance-metrics.html', title=title)

# ===============================================================================================

# RENDER PREDICTION PAGES

# render crop recommendation result page


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Precision Farming - Crop Recommendation'

    if request.method == 'POST':
        N = float(request.form['nitrogen'])
        P = float(request.form['phosphorous'])
        K = float(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction_encoded = crop_recommendation_model.predict(data)
            prediction_decoded = le.inverse_transform(prediction_encoded)
            final_prediction = prediction_decoded[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)
# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
