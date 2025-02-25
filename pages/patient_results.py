import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(initial_sidebar_state="collapsed") 
st.markdown( """ <style> [data-testid="collapsedControl"] { display: none } </style> """, unsafe_allow_html=True, )

st.title("Patient Results")
st.write("Based on the data you inputted, the patient results are dispalyed and analyzed below.")

st.write(type(st.session_state.input_data[0]))
model = st.session_state.model
output_index = model.get_output_details()[0]["index"]
model.set_tensor(0, st.session_state.input_data)
model.invoke()
output_data = model.get_tensor(output_index[0]['index'])
predicted_class = np.argmax(output_data)
st.write("I predict that the patient belongs to class: ", predicted_class)


