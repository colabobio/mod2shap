import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(initial_sidebar_state="collapsed") 
st.markdown( """ <style> [data-testid="collapsedControl"] { display: none } </style> """, unsafe_allow_html=True, )

st.title('New Patient Info')
st.write("Below is the patient form we created based on your data. Please fill it out with your new patient data to create new predicitions and visualizations!")
data = st.session_state.data
feature_names = data.columns[:-1]
data_types = data.dtypes.values

input_data = []
with st.form(key="new_patient_form"):
    for i in range (len(feature_names)):
        #int or float input value
        if ('int' in str(data_types[i])):
            patient_data = st.number_input(label=feature_names[i])
            input_data.append(np.float32(patient_data))
        #boolean input value
        elif ('bool' in str(data_types[i])):
            patient_data = st.checkbox(label=feature_names[i])
            input_data.append(eval(patient_data))
        elif ('category' in str(data_types[i])):
            patient_data = st.selectbox(label=feature_names[i], options=data[feature_names[i]].unique())
            input_data.append(patient_data)
        else:
            patient_data = st.text_input(label=feature_names[i])
            input_data.append(patient_data)
    submit = submit_button=st.form_submit_button(label="Submit", type='primary')

if submit:
    st.session_state.input_data = input_data
    st.switch_page("./pages/patient_results.py")
