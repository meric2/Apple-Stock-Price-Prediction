import subprocess
import sys

#def install(package):
 #   subprocess.check_call([sys.executable, "-m", "pip", "install", package])

#install("lightgbm")

import streamlit as st
import pandas as pd
import pickle
from PIL import Image
from xgboost import XGBRegressor


st.write("""
# Apple Stock Price Prediction

This app predicts the adjusted close value of the given day within the other daily values.

""")


st.sidebar.header('Daily Inputs')


def user_input_features():
    op = st.sidebar.slider('Open', -1.0, 0.99, 0.01)
    high = st.sidebar.slider('High', -1.0, 0.99, 0.01)
    low = st.sidebar.slider('Low', 0, 100, 1)
    close = st.sidebar.slider('Close', 41, 160, 1)
    vol = st.sidebar.slider('Volume', 41, 160, 1)
    data = {'Open': op,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': vol}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('Daily Inputs')
st.write(df)

model = pickle.load(open("model.pkl", 'rb'))
prediction = model.predict(df)

st.subheader('Prediction Result:' + prediction)

#image = Image.open('photo.jpg')
#st.image(image)
