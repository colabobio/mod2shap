import streamlit as st
import pandas as pd
import tensorflow as tf
from tempfile import NamedTemporaryFile

st.set_page_config(initial_sidebar_state="collapsed") 
st.markdown( """ <style> [data-testid="collapsedControl"] { display: none } </style> """, unsafe_allow_html=True, )

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
        st.session_state.model = interpreter

st.header("Data")
st.write("Upload your training data below. Please make sure that the first row contains feature and label names, with the last column holding the label information.")
uploaded_data = st.file_uploader("Upload data", type=[".csv"])
if (uploaded_data != None):
    data_df = pd.read_csv(uploaded_data)
    st.session_state.data = data_df
    preview_training = data_df.head()
    st.table(preview_training)

st.write("")
st.write("")
generate_patient_form = st.button("Generate patient form", type='primary')
if generate_patient_form:
    if (uploaded_model == None):
        st.write(":red[Please upload your training data before proceeding!]")
    elif (uploaded_data == None):
        st.write(":red[Please upload your data before proceeeding!]")
    else:
        st.switch_page("./pages/patient_form.py")