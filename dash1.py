import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import requests
import os
import ee
from datetime import datetime

# --- DIRECTORY CONFIGURATION ---
# Hardcoded to your specific paths to eliminate any local environment confusion
MODELS_DIR = r"D:\FINALYEARPROJECT\FORESTFIREPREDICTION\models"
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forest Fire Prediction System", page_icon="🌲", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size: 22px !important; font-weight: bold; }
    .alert-extreme { background-color: #8c564b; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .alert-high { background-color: #d62728; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .alert-moderate { background-color: #ff7f0e; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .alert-low { background-color: #2ca02c; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("🌲 Forest Fire Predictive Dashboard")
st.markdown("Automated risk assessment using Google Earth Engine and Open-Meteo telemetry.")

# --- INITIALIZE EARTH ENGINE ---
def init_ee():
    try:
        key_file_path = os.path.join(DASHBOARD_DIR, 'ee-key.json')
        if not os.path.exists(key_file_path):
             return False, f"Key file not found at: {key_file_path}"
             
        credentials = ee.ServiceAccountCredentials('', key_file_path)
        ee.Initialize(credentials, project='lofty-inn-490212-m9')
        # Test connection with a small request
        ee.Image("USGS/SRTMGL1_003").getMapId()
        return True, None
    except Exception as e:
        return False, str(e)

ee_ready, ee_error = init_ee()

# --- LOAD MODELS ---
def load_models():
    models = {}
    
    # Using the hardcoded path you confirmed
    ACTUAL_MODELS_DIR = r"D:\FINALYEARPROJECT\FORESTFIREPREDICTION\models"
    
    model_files = {
        'Random Forest': 'forest_fire_rf_model.pkl',
        'XGBoost': 'forest_fire_xgb_model.pkl',
        'MLP Neural Net': 'forest_fire_mlp_model.pkl',
        'Hybrid Ensemble': 'forest_fire_hybrid_model.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(ACTUAL_MODELS_DIR, filename)
        try:
            # THE FIX: Use joblib.load instead of pickle.load
            models[name] = joblib.load(path)
        except FileNotFoundError:
            st.error(f"⚠️ Missing: {filename} in {ACTUAL_MODELS_DIR}")
        except Exception as e:
            st.error(f"⚠️ Error loading {name}: {e}")
            
    return models

# --- DATA FETCHING FUNCTIONS ---
def get_gee_data(lat, lon):
    try:
        point = ee.Geometry.Point([lon, lat])
        srtm = ee.Image('USGS/SRTMGL1_003')
        elev = srtm.reduceRegion(ee.Reducer.mean(), point, 30).get('elevation').getInfo()
        
        terrain = ee.Terrain.products(srtm)
        slope = terrain.select('slope').reduceRegion(ee.Reducer.mean(), point, 30).get('slope').getInfo()
        aspect = terrain.select('aspect').reduceRegion(ee.Reducer.mean(), point, 30).get('aspect').getInfo()
        
        modis = ee.ImageCollection('MODIS/061/MOD13Q1').filterBounds(point).sort('system:time_start', False).first()
        ndvi_raw = modis.select('NDVI').reduceRegion(ee.Reducer.mean(), point, 250).get('NDVI').getInfo()
        
        return (
            float(elev) if elev else 1500.0,
            float(slope) if slope else 5.0,
            float(aspect) if aspect else 180.0,
            (float(ndvi_raw) * 0.0001) if ndvi_raw else 0.35
        )
    except:
        return 1500.0, 5.0, 180.0, 0.35

def get_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_max&past_days=7&forecast_days=1&timezone=auto"
    try:
        r = requests.get(url).json()['daily']
        # Current (Day 7) and Averages (Days 0-6)
        return (
            r['temperature_2m_max'][7], 
            r['relative_humidity_2m_max'][7], 
            r['wind_speed_10m_max'][7], 
            r['precipitation_sum'][7], 
            np.mean(r['temperature_2m_max'][:7]), 
            np.sum(r['precipitation_sum'][:7])
        )
    except:
        return None

# --- SIDEBAR CONTROLS ---
st.sidebar.header("📍 Location Parameters")
latitude = st.sidebar.number_input("Latitude", value=30.1459, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=78.7672, format="%.4f")

# --- MAIN LOGIC ---
if not ee_ready:
    st.error(f"🛑 Earth Engine Initialization Failed: {ee_error}")
else:
    st.sidebar.success("✅ Systems Online")

    if st.sidebar.button("Run Prediction", type="primary"):
        with st.spinner("Extracting environmental telemetry..."):
            
            gee = get_gee_data(latitude, longitude)
            weath = get_weather_data(latitude, longitude)
            
            if weath:
                max_t, hum, wind, rain, t7_avg, r7_sum = weath
                danger = (max_t * wind) / (hum + 1)
                month = datetime.now().month
                
                # ==========================================================
                # THE FIX: EXACT FEATURE NAMES AND 1-ROW LIST FORMAT
                # ==========================================================
                live_input = pd.DataFrame({
                    'Month': [float(month)],
                    'Rain_7d_Sum': [float(r7_sum)],
                    'Temp_7d_Avg': [float(t7_avg)],
                    'Fire_Danger_Index': [float(danger)],
                    'Max_Temperature_C': [float(max_t)],
                    'Max_Humidity_pct': [float(hum)],
                    'Total_Rainfall_mm': [float(rain)],
                    'Max_Wind_Speed_kmh': [float(wind)],
                    'Elevation_m': [float(gee[0])],
                    'Slope_deg': [float(gee[1])],
                    'Aspect_deg': [float(gee[2])],
                    'Baseline_NDVI': [float(gee[3])]
                })

                st.subheader("Data Extraction Results")
                d_cols = st.columns(4)
                d_cols[0].metric("Elevation", f"{gee[0]:.0f}m")
                d_cols[1].metric("7D Rain Sum", f"{r7_sum:.1f}mm")
                d_cols[2].metric("Temperature", f"{max_t:.1f}°C")
                d_cols[3].metric("NDVI", f"{gee[3]:.3f}")
                st.markdown("---")

                models = load_models()
                
                if models and 'Hybrid Ensemble' in models:
                    risk_map = {0: 'Low Risk', 1: 'Moderate Risk', 2: 'High Risk', 3: 'Extreme Risk'}
                    
                    try:
                        # Converting to raw array to prevent Panda Index errors
                        input_matrix = live_input.values
                        
                        final_pred = models['Hybrid Ensemble'].predict(input_matrix)[0]
                        
                        if final_pred == 3:
                            st.markdown('<div class="alert-extreme"><h1>🚨 EXTREME RISK</h1><p class="big-font">Conditions are critical. Immediate warning recommended.</p></div>', unsafe_allow_html=True)
                        elif final_pred == 2:
                            st.markdown('<div class="alert-high"><h1>⚠️ HIGH RISK</h1><p class="big-font">High danger of forest fire ignition.</p></div>', unsafe_allow_html=True)
                        elif final_pred == 1:
                            st.markdown('<div class="alert-moderate"><h1>🔔 MODERATE RISK</h1><p class="big-font">Monitor the area for smoke or fire signs.</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-low"><h1>✅ LOW RISK</h1><p class="big-font">No immediate fire threat detected.</p></div>', unsafe_allow_html=True)

                        st.write("### AI Component Breakdown")
                        m_cols = st.columns(4)
                        for i, (name, model) in enumerate(models.items()):
                            m_pred = model.predict(input_matrix)[0]
                            m_cols[i].info(f"**{name}**\n\n{risk_map[m_pred]}")
                            
                    except Exception as e:
                         st.error(f"Prediction Error: {e}")