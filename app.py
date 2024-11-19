import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Import model and data
st.title("Laptop Price Predictor 💻")
pipe = pickle.load(open(r"C:\Users\Vaibhav\PycharmProjects\Laptop_price_prediction\pipe1.pkl", "rb"))
df = pickle.load(open(r"C:\Users\Vaibhav\PycharmProjects\Laptop_price_prediction\df.pkl", "rb"))

# User inputs
left_column, middle_column, right_column = st.columns(3)
with left_column:
    company = st.selectbox("Brand", df["Company"].unique())
with middle_column:
    type = st.selectbox("Type", df["TypeName"].unique())
with right_column:
    ram = st.selectbox("Ram (in GB)", df["Ram"].unique())

left_column, middle_column, right_column = st.columns(3)
with left_column:
    weight = st.number_input("Weight of laptop in kg")
with middle_column:
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])
with right_column:
    ips = st.selectbox("IPS Display", ["No", "Yes"])

left_column, middle_column, right_column = st.columns(3)
with left_column:
    Screen_size = st.number_input("Screen Size (in Inches)")
with middle_column:
    resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
with right_column:
    cpu = st.selectbox("CPU Brand", df["Cpu brand"].unique())

left_column, right_column = st.columns(2)
with left_column:
    hdd = st.selectbox("HDD(in GB)", [0, 128, 256, 512, 1024, 2048])
with right_column:
    ssd = st.selectbox("SSD(in GB)", [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox("GPU Brand", df["Gpu brand"].unique())
os = st.selectbox("OS Type", df["os"].unique())

if st.button("Predict Price"):
    # Convert input features to numeric
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    try:
        X_res = int(resolution.split("x")[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / Screen_size
    except ZeroDivisionError:
        st.error("Screen Size must be greater than 0")
        ppi = 0  # Or handle this case as appropriate

    # Create a DataFrame with the same structure as the model was trained on
    input_data = pd.DataFrame({
        'Company': [company],
        'TypeName': [type],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'Ips': [ips],
        'ppi': [ppi],
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })

    try:
        prediction = pipe.predict(input_data)[0]
        st.title(f"The Predicted Price of Laptop = Rs {int(np.exp(prediction))}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
