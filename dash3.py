import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import ee
from datetime import datetime

# --- DIRECTORY CONFIGURATION ---
MODELS_DIR = r"D:\FINALYEARPROJECT\LANDSLIDEPREDICTOR\models"
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Landslide Early Warning", page_icon="⛰️", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size: 22px !important; font-weight: bold; }
    .alert-extreme { background-color: #8c564b; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .alert-high { background-color: #d62728; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    .alert-low { background-color: #2ca02c; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.title("⛰️ Landslide Risk Analysis Dashboard")
st.markdown("Geospatial terrain assessment and antecedent moisture tracking system.")

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

# --- LOAD MODELS & SCALER ---
@st.cache_resource
def get_live_scaler():
    """Loads the pre-fitted scaler.pkl found by the diagnostic."""
    try:
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"⚠️ Could not load 'scaler.pkl'. Error: {e}")
        return None

def load_models():
    models = {}
    
    # EXACT FILENAMES from your diagnostic output
    model_files = {
        'Random Forest': 'random_forest_landslide_model.pkl',
        'XGBoost': 'xgboost_landslide_model.pkl',
        'SVM': 'svm_landslide_model.pkl',
        'Hybrid Stacking': 'hybrid_landslide_model.pkl' 
    }
    
    if not os.path.exists(MODELS_DIR):
        st.error(f"❌ Directory NOT found: {MODELS_DIR}")
        return models

    for name, filename in model_files.items():
        path = os.path.join(MODELS_DIR, filename)
        try:
            models[name] = joblib.load(path)
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
        
        modis = ee.ImageCollection('MODIS/061/MOD13Q1').filterBounds(point).sort('system:time_start', False).first()
        ndvi_raw = modis.select('NDVI').reduceRegion(ee.Reducer.mean(), point, 250).get('NDVI').getInfo()
        
        return (float(elev) if elev else 1500.0, float(slope) if slope else 5.0, float(aspect) if aspect else 180.0, (float(ndvi_raw) * 0.0001) if ndvi_raw else 0.35)
    except:
        return 1500.0, 5.0, 180.0, 0.35

def get_weather_data(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum,soil_moisture_0_to_7cm_mean,soil_moisture_28_to_100cm_mean&past_days=15&forecast_days=1&timezone=auto"
    try:
        r = requests.get(url).json()['daily']
        rain = r['precipitation_sum']
        
        rain_0d = rain[15]
        rain_3d = np.sum(rain[12:15])
        rain_7d = np.sum(rain[8:15])
        rain_15d = np.sum(rain[0:15])
        
        soil_surf = r['soil_moisture_0_to_7cm_mean'][15]
        soil_deep = r['soil_moisture_28_to_100cm_mean'][15]
        
        return rain_0d, rain_3d, rain_7d, rain_15d, soil_surf, soil_deep
    except:
        return 0, 0, 0, 0, 0.2, 0.2

# --- SIDEBAR ---
st.sidebar.header("📍 Topographical Target")
latitude = st.sidebar.number_input("Latitude", value=30.1459, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=78.7672, format="%.4f")

# --- EXECUTION ---
if not ee_ready:
    st.error(f"🛑 EE Initialization Failed: {ee_error}")
else:
    if st.sidebar.button("Run Slope Stability Analysis", type="primary"):
        with st.spinner("Extracting multi-spectral and geological telemetry..."):
            
            gee = get_gee_data(latitude, longitude)
            weath = get_weather_data(latitude, longitude)
            scaler = get_live_scaler()
            
            if weath and scaler:
                elev, slope, aspect, ndvi = gee
                rain_0d, rain_3d, rain_7d, rain_15d, soil_surf, soil_deep = weath
                month = datetime.now().month
                
                # --- ENGINEERED FEATURES ---
                month_sin = np.sin(2 * np.pi * month / 12)
                month_cos = np.cos(2 * np.pi * month / 12)
                aspect_sin = np.sin(np.radians(aspect))
                aspect_cos = np.cos(np.radians(aspect))
                soil_grad = soil_deep - soil_surf
                rain_slope = rain_7d * slope
                water_load = rain_15d + soil_deep
                
                r_norm = rain_0d / 100.0
                s_norm = slope / 90.0
                m_norm = soil_surf
                risk_score = (rain_7d * slope) / 100.0
                noise = np.random.rand()

                live_input = pd.DataFrame({
                    'Elevation_m': [elev],
                    'Slope_deg': [slope],
                    'Baseline_NDVI': [ndvi],
                    'Rainfall_Day_0_mm': [rain_0d],
                    'Rainfall_Antecedent_3D_mm': [rain_3d],
                    'Rainfall_Antecedent_7D_mm': [rain_7d],
                    'Rainfall_Antecedent_15D_mm': [rain_15d],
                    'Soil_Moisture_Surface': [soil_surf],
                    'Soil_Moisture_Deep': [soil_deep],
                    'Month_Sin': [month_sin],
                    'Month_Cos': [month_cos],
                    'Aspect_Sin': [aspect_sin],
                    'Aspect_Cos': [aspect_cos],
                    'Soil_Moisture_Gradient': [soil_grad],
                    'Rain_Slope_Interaction': [rain_slope],
                    'Total_15D_Water_Load': [water_load],
                    'Rain_norm': [r_norm],
                    'Slope_norm': [s_norm],
                    'Moisture_norm': [m_norm],
                    'Risk_Score': [risk_score],
                    'random_noise_feature': [noise]
                })

                st.subheader("Geospatial & Antecedent Profile")
                d_cols = st.columns(4)
                d_cols[0].metric("Slope Gradient", f"{slope:.1f}°")
                d_cols[1].metric("15-Day Rain", f"{rain_15d:.1f}mm")
                d_cols[2].metric("Deep Soil", f"{soil_deep:.2f}")
                d_cols[3].metric("NDVI", f"{ndvi:.2f}")
                st.markdown("---")

                models = load_models()
                
                if models and 'Hybrid Stacking' in models:
                    risk_map = {0: 'Stable', 1: 'Moderate Risk', 2: 'High Risk'}
                    
                    try:
                        # 1. Extract pure matrix
                        input_matrix = live_input.values
                        
                        # 2. Scale using the actual scaler.pkl
                        input_scaled = scaler.transform(input_matrix)
                        
                        # 3. Predict
                        final_pred = models['Hybrid Stacking'].predict(input_scaled)[0]
                        
                        if final_pred == 2:
                            st.markdown('<div class="alert-extreme"><h1>🚨 HIGH LANDSLIDE RISK</h1><p class="big-font">Critical slope instability detected.</p></div>', unsafe_allow_html=True)
                        elif final_pred == 1:
                            st.markdown('<div class="alert-high"><h1>⚠️ MODERATE RISK</h1><p class="big-font">Elevated antecedent moisture. Monitor area.</p></div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="alert-low"><h1>✅ STABLE CONDITIONS</h1><p class="big-font">Minimal mass-movement probability.</p></div>', unsafe_allow_html=True)

                        st.write("### Consensus Breakdown")
                        m_cols = st.columns(len(models))
                        for i, (name, model) in enumerate(models.items()):
                            m_pred = model.predict(input_scaled)[0]
                            m_cols[i].info(f"**{name}**\n\n{risk_map[m_pred]}")
                            
                    except Exception as e:
                         st.error(f"Analysis Crash: {e}")