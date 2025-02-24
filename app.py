import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile

st.title('Mod2Shap')
st.write("Welcome! Mod2Shap is a web application that can help you run your custom ML models on new patient data and visualize your model's predictions.")
st.write("")

st.header("Model")
st.write("Upload your .tflite model below.")
uploaded_model = st.file_uploader("Upload Model", type=["tflite"])
if (uploaded_model != None):
    with NamedTemporaryFile(delete=False) as temp:
        temp.write(uploaded_model.read())
        model_path = temp.name
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

st.header("Data")
st.write("Upload your training data below. Please make sure it is in the form of a csv file")
uploaded_data = st.file_uploader("Upload data", type=[".csv"])
if (uploaded_data != None):
    data_df = pd.read_csv(uploaded_data)
    preview_training = data_df.head()
    st.table(preview_training)

st.write("")
st.write("")
st.button("Generate patient form", type='primary')