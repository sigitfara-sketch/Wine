import streamlit as st
import pandas as pd
import numpy as np
import joblib


st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="centered"
)


@st.cache_resource
def load_model():
    model = joblib.load("wine_quality_model.joblib")
    scaler = joblib.load("wine_quality_scaler.joblib")
    return model, scaler

model, scaler = load_model()

st.markdown(
    """
    <h1 style="text-align:center; color:#8B0000;">🍷 Wine Quality Prediction</h1>
    <p style="text-align:center;">
    Prediksi kualitas wine menggunakan <b>Random Forest Classifier</b>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)


st.sidebar.header("🧪 Input Karakteristik Wine")

def user_input():
    fixed_acidity = st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 7.4)
    volatile_acidity = st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.7)
    citric_acid = st.sidebar.slider("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.sidebar.slider("Residual Sugar", 0.5, 15.0, 1.9)
    chlorides = st.sidebar.slider("Chlorides", 0.01, 0.6, 0.076)
    free_sulfur_dioxide = st.sidebar.slider("Free Sulfur Dioxide", 1, 80, 11)
    total_sulfur_dioxide = st.sidebar.slider("Total Sulfur Dioxide", 6, 300, 34)
    density = st.sidebar.slider("Density", 0.9900, 1.0050, 0.9978)
    pH = st.sidebar.slider("pH", 2.5, 4.5, 3.51)
    sulphates = st.sidebar.slider("Sulphates", 0.3, 2.0, 0.56)
    alcohol = st.sidebar.slider("Alcohol (%)", 8.0, 15.0, 9.4)

    data = {
        "fixed acidity": fixed_acidity,
        "volatile acidity": volatile_acidity,
        "citric acid": citric_acid,
        "residual sugar": residual_sugar,
        "chlorides": chlorides,
        "free sulfur dioxide": free_sulfur_dioxide,
        "total sulfur dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol
    }

    return pd.DataFrame([data])

input_df = user_input()


st.subheader("📋 Data Input Wine")
st.dataframe(input_df, use_container_width=True)


if st.button("🔍 Prediksi Kualitas Wine"):
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)

    st.subheader("📊 Hasil Prediksi")

    if prediction == 1:
        st.success("✅ **Wine Berkualitas Tinggi (Good Quality)**")
    else:
        st.error("❌ **Wine Berkualitas Rendah (Low Quality)**")

    st.markdown("**Probabilitas Prediksi:**")
    prob_df = pd.DataFrame(
        probability,
        columns=["Low Quality", "Good Quality"]
    )
    st.bar_chart(prob_df.T)


st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:12px;">
    Dibuat dengan ❤️ menggunakan Streamlit & Random Forest
    </p>
    """,
    unsafe_allow_html=True
)
