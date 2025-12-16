import streamlit as st
import pandas as pd
import joblib
import os

# ===============================
# KONFIGURASI HALAMAN
# ===============================
st.set_page_config(
    page_title="Prediksi Kualitas Anggur Merah",
    page_icon="🍷",
    layout="wide"
)

# ===============================
# JUDUL
# ===============================
st.title("🍷 Prediksi Kualitas Anggur Merah")
st.markdown("""
Aplikasi ini menggunakan **Random Forest Classifier**
(dalam bentuk Pipeline: Scaler + Model)
untuk memprediksi kualitas anggur merah:
- **1 = Kualitas Tinggi (≥ 7)**
- **0 = Kualitas Rendah (< 7)**
""")
st.divider()

# ===============================
# LOAD PIPELINE (SATU FILE)
# ===============================
PIPELINE_PATH = "wine_quality_pipeline.joblib"

@st.cache_resource
def load_pipeline():
    if not os.path.exists(PIPELINE_PATH):
        st.error("❌ File `wine_quality_pipeline.joblib` tidak ditemukan di repository GitHub.")
        st.stop()
    return joblib.load(PIPELINE_PATH)

pipeline = load_pipeline()

feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid',
    'residual sugar', 'chlorides', 'free sulfur dioxide',
    'total sulfur dioxide', 'density', 'pH',
    'sulphates', 'alcohol'
]

# ===============================
# INPUT USER
# ===============================
st.header("🔧 Masukkan Parameter Anggur")

feature_ranges = {
    'fixed acidity': (4.0, 16.0, 8.3),
    'volatile acidity': (0.1, 1.6, 0.5),
    'citric acid': (0.0, 1.0, 0.3),
    'residual sugar': (0.9, 15.5, 2.5),
    'chlorides': (0.01, 0.65, 0.08),
    'free sulfur dioxide': (1.0, 72.0, 15.8),
    'total sulfur dioxide': (6.0, 289.0, 46.4),
    'density': (0.99, 1.004, 0.996),
    'pH': (2.7, 4.0, 3.3),
    'sulphates': (0.3, 2.0, 0.66),
    'alcohol': (8.4, 14.9, 10.4)
}

cols = st.columns(4)
input_data = {}

for i, feature in enumerate(feature_names):
    min_v, max_v, default = feature_ranges[feature]
    with cols[i % 4]:
        input_data[feature] = st.slider(
            feature.title(),
            min_value=min_v,
            max_value=max_v,
            value=default
        )

# ===============================
# PREDIKSI
# ===============================
st.divider()
if st.button("🔬 Prediksi Kualitas Anggur"):
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]

    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]

    st.subheader("🎯 Hasil Prediksi")

    if prediction == 1:
        st.success("✅ **Kualitas Anggur: TINGGI**")
        st.balloons()
    else:
        st.error("❌ **Kualitas Anggur: RENDAH**")

    st.info(f"""
    **Probabilitas Rendah:** {proba[0]*100:.2f}%  
    **Probabilitas Tinggi:** {proba[1]*100:.2f}%
    """)

# ===============================
# FEATURE IMPORTANCE
# ===============================
st.divider()
st.header("📊 Feature Importance")

try:
    import plotly.express as px

    model = pipeline.named_steps["model"]

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Pengaruh Fitur terhadap Prediksi"
    )

    st.plotly_chart(fig, use_container_width=True)

except Exception:
    st.warning("Plotly belum terpasang. Tambahkan `plotly` ke requirements.txt jika ingin grafik.")

st.caption("© Streamlit • Random Forest Pipeline • Python 3.10")
