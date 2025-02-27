import streamlit as st
import numpy as np

st.set_page_config(initial_sidebar_state="collapsed")
st.markdown("""<style> [data-testid="collapsedControl"] { display: none } </style>""", unsafe_allow_html=True)

st.title("Patient Results")
st.write("Based on your input, the patient results are displayed below.")

model = st.session_state.model
expected_features = model.get_input_details()[0]['shape'][1]

# Trim extra values or add 0.0s to match expected feature count 
# If expected features is 4 but input before is [2.0, 2.13, 5.5] then it will append 0.0 to make it 4 features --- trims if too many values 
st.session_state.input_data = (st.session_state.input_data[:expected_features] + [0.0] * (expected_features - len(st.session_state.input_data)))

# Run inference
model.set_tensor(0, np.array([st.session_state.input_data], dtype=np.float32)) #Reshapes to (1, num_features)
model.invoke()
predicted_class = np.argmax(model.get_tensor(model.get_output_details()[0]["index"]))

st.write(f"The Patient belongs to class: {['No', 'Yes'][predicted_class]}")


if st.button("Generate SHAP Visualization", type='primary'):
    st.switch_page('./pages/shap_visualization.py')