import streamlit as st
import pandas as pd

st.set_page_config(initial_sidebar_state="collapsed") 
st.markdown( """ <style> [data-testid="collapsedControl"] { display: none } </style> """, unsafe_allow_html=True, )

st.title('New Patient Info')
st.write("Below is the patient form we created based on your data. Please fill it out with your new patient data to create new predicitions and visualizations!")
st.write(st.session_state.data)