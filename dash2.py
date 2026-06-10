import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os
import ee
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation
# from streamlit_geolocation import streamlit_geolocation
# --- DIRECTORY CONFIGURATION ---
MODELS_DIR = r"D:\FINALYEARPROJECT\CLOUDBURSTHEAVYRAINFALL\models"
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Cloudburst Early Warning", page_icon="⛈️", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size: 22px !important; font-weight: bold; }
    .alert-extreme { background-color: #4b0082; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .alert-high { background-color: #ff4500; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .alert-low { background-color: #2ca02c; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("⛈️ Cloudburst Risk Analysis Dashboard")
st.markdown("Automated prediction system for extreme precipitation events in the Himalayan region.")

# --- INITIALIZE EARTH ENGINE ---
def init_ee():
    try:
        key_file_path = os.path.join(DASHBOARD_DIR, 'ee-key.json')
        if not os.path.exists(key_file_path):
             return False, f"Key file not found at: {key_file_path}"

        credentials = ee.ServiceAccountCredentials('', key_file_path)
        ee.Initialize(credentials, project='lofty-inn-490212-m9')
        return True, None
    except Exception as e:
        return False, str(e)

ee_ready, ee_error = init_ee()

# --- LOAD MODELS (Aligned with your actual folder filenames) ---
def load_models():
    models = {}
    model_files = {
        'RF Model': 'model_rf.pkl',
        'XGB Model': 'model_xgb.pkl',
        'SVM Model': 'model_svm.pkl',
        'Tuned Hybrid': 'model_hybrid_tuned.pkl'
    }

    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        try:
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        except Exception as e:
            st.error(f"⚠️ Could not load {filename}: {e}")
    return models

# --- DATA FETCHING ---
def get_gee_data(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        srtm = ee.Image('USGS/SRTMGL1_003')
        elev = srtm.reduceRegion(ee.Reducer.mean(), point, 30).get('elevation').getInfo()
        terrain = ee.Terrain.products(srtm)
        slope = terrain.select('slope').reduceRegion(ee.Reducer.mean(), point, 30).get('slope').getInfo()
        aspect = terrain.select('aspect').reduceRegion(ee.Reducer.mean(), point, 30).get('aspect').getInfo()
        return (float(elev) if elev else 1500.0, float(slope) if slope else 5.0, float(aspect) if aspect else 180.0)
    except:
        return 1500.0, 5.0, 180.0

def get_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum,relative_humidity_2m_max&past_days=7&forecast_days=1&timezone=auto"
    try:
        r = requests.get(url).json()['daily']
        return (
            r['temperature_2m_max'][7],
            r['relative_humidity_2m_max'][7],
            r['precipitation_sum'][7],
            np.mean(r['temperature_2m_max'][:7]),
            np.sum(r['precipitation_sum'][:7])
        )
    except:
        return None

# --- MANUAL SIDEBAR ---
# st.sidebar.header("📍 Targeting Coordinates")
# latitude = st.sidebar.number_input("Latitude", value=30.1459, format="%.4f")
# longitude = st.sidebar.number_input("Longitude", value=78.7672, format="%.4f")

# --- REAL TIME SIDEBAR ---
st.sidebar.header("📍 Current Location")



location = streamlit_geolocation()

if location and location["latitude"] is not None:
    latitude = location["latitude"]
    longitude = location["longitude"]

    st.sidebar.success("Location detected")
    st.sidebar.write(f"Lat: {latitude:.6f}")
    st.sidebar.write(f"Lon: {longitude:.6f}")

else:
    st.sidebar.warning("Allow location access in your browser")
    latitude = 30.1459
    longitude = 78.7672


# --- EXECUTION ---
if not ee_ready:
    st.error(f"🛑 EE Initialization Failed: {ee_error}")
else:
    if st.sidebar.button("Run Risk Assessment", type="primary"):
        with st.spinner("Extracting environmental telemetry..."):

            gee = get_gee_data(latitude, longitude)
            weath = get_weather_data(latitude, longitude)

            if weath:
                max_t, hum, rain, t7_avg, r7_sum = weath
                month = datetime.now().month

                # ==========================================================
                # THE 9-FEATURE FIX (Matches model training exactly)
                # ==========================================================
                live_input = pd.DataFrame({
                    'Month': [float(month)],
                    'Rain_7d_Sum': [float(r7_sum)],
                    'Temp_7d_Avg': [float(t7_avg)],
                    'Max_Temperature_C': [float(max_t)],
                    'Max_Humidity_pct': [float(hum)],
                    'Total_Rainfall_mm': [float(rain)],
                    'Elevation_m': [float(gee[0])],
                    'Slope_deg': [float(gee[1])],
                    'Aspect_deg': [float(gee[2])]
                })

                st.subheader("Current Environmental State")
                d_cols = st.columns(3)
                d_cols[0].metric("Elevation", f"{gee[0]:.0f}m")
                d_cols[1].metric("7D Rainfall", f"{r7_sum:.1f}mm")
                d_cols[2].metric("Humidity", f"{hum:.0f}%")
                st.markdown("---")

                models = load_models()

                if models and 'Tuned Hybrid' in models:
                    risk_map = {0: 'Stable/Normal Weather', 1: 'Heavy Rainfall Risk', 2: 'Cloudburst Warning'}

                    try:
                        # Pass raw numeric matrix to avoid indexing errors
                        input_matrix = live_input.values

                        final_pred = models['Tuned Hybrid'].predict(input_matrix)[0]

                        if final_pred == 2:
                            st.markdown('<div class="alert-extreme"><h1>🚨 CLOUDBURST WARNING</h1><p class="big-font">Extreme atmospheric instability. Precautionary protocols recommended.</p></div>', unsafe_allow_html=True)
                        elif final_pred == 1:
                            st.markdown('<div class="alert-high"><h1>⚠️ HEAVY RAINFALL RISK</h1><p class="big-font">High volume precipitation expected. Monitor drainage capacity.</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-low"><h1>✅ STABLE CONDITIONS</h1><p class="big-font">No immediate cloudburst threat detected.</p></div>', unsafe_allow_html=True)

                        st.write("### Consensus Breakdown")
                        m_cols = st.columns(len(models))
                        for i, (name, model) in enumerate(models.items()):
                            m_pred = model.predict(input_matrix)[0]
                            m_cols[i].info(f"**{name}**\n\n{risk_map[m_pred]}")

                    except Exception as e:
                         st.error(f"Analysis Crash: {e}")