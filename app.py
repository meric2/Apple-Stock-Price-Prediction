import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("xgboost")

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
    op = st.sidebar.text_input('Open')
    high = st.sidebar.text_input('High')
    low = st.sidebar.text_input('Low')
    close = st.sidebar.text_input('Close')
    vol = st.sidebar.text_input('Volume')
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
