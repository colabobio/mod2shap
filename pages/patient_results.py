import streamlit as st
import numpy as np

st.set_page_config(initial_sidebar_state="collapsed")
st.markdown("""<style> [data-testid="collapsedControl"] { display: none } </style>""", unsafe_allow_html=True)

st.title("Patient Results")
st.write("Based on your input, the patient results are displayed below.")

model = st.session_state.model
input = st.session_state.input_data
input_details = model.get_input_details()
output_details = model.get_output_details()

if len(input_details > 1):
    for input_detail in input_details:
        name = input_detail['name']
        v = name[name.find('input_feats')+12:-2]
        idx = v.split('-')
        if len(idx) == 1:
            data = input[int(idx[0]):int(idx[0])+1]
        elif len(idx) == 2:
            data = input[int(idx[0]):int(idx[1])+1]

        input_shape = input_detail['shape']
        input_type = input_detail['dtype']
        input_shaped_data = np.array(data).reshape(input_shape).astype(input_type)
        model.set_tensor(input_detail['index'], input_shaped_data)

else:
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']
    input_shaped_data = np.array(input).reshape(input_shape).astype(input_type)
    model.set_tensor(input_details[0]['index'], input_shaped_data)
        
model.invoke()
output_data = model.get_tensor(output_details[0]['index'])
np.array(output_data.flatten())  # Flatten the output for Shap
st.write(f"Prediction: ", str(output_data[0][0] * 100) + "%")

if st.button("Generate SHAP Visualization", type='primary'):
    st.switch_page('./pages/shap_visualization.py')