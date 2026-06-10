import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import os
import ee
from datetime import datetime
from streamlit_geolocation import streamlit_geolocation
from twilio.rest import Client

# Defining the sos route

def send_sos_alert(latitude, longitude, alerts: list):
    try:
        client = Client(
            st.secrets["TWILIO_ACCOUNT_SID"],
            st.secrets["TWILIO_AUTH_TOKEN"]
        )

        alert_lines = "\n".join(
            [f"🔴 {a['hazard']}: {a['level']} — {a['detail']}" for a in alerts]
        )

        # --- Fetch safe zones for the message ---
        api_key = st.secrets["GEOAPIFY_API_KEY"]
        safezone_lines = ""
        maps_search_url = ""

        try:
            # Determine search category based on hazard
            hazard_names = [a["hazard"] for a in alerts]
            if any("FIRE" in h for h in hazard_names):
                category = "healthcare.hospital,service.fire_station,service.police"
                search_term = "hospital+fire+station+near"
            elif any("RAIN" in h or "CLOUD" in h for h in hazard_names):
                category = "healthcare.hospital,education.school,activity.community_center,service.police"
                search_term = "hospital+shelter+near"
            else:
                category = "healthcare.hospital,service.police,education.school"
                search_term = "hospital+police+near"
            url = (
                f"https://api.geoapify.com/v2/places"
                f"?categories={category}"
                f"&filter=circle:{longitude},{latitude},30000"
                f"&limit=5"
                f"&apiKey={api_key}"
            )
            r = requests.get(url, timeout=10).json()
            places = []
            for feature in r.get("features", []):
                props = feature.get("properties", {})
                coords = feature.get("geometry", {}).get("coordinates", [None, None])
                name = props.get("name") or props.get("address_line1") or "Safe Zone"
                dist = round(props.get("distance", 0) / 1000, 1)
                p_lat, p_lon = coords[1], coords[0]
                if name and p_lat and p_lon:
                    places.append({
                        "name": name,
                        "lat": p_lat,
                        "lon": p_lon,
                        "dist": dist
                    })

            if places:
                # Build safe zone lines for SMS
                safezone_lines = "\n🛡️ NEAREST SAFE ZONES:\n"
                for i, p in enumerate(places[:5], 1):
                    nav_url = f"https://www.google.com/maps/dir/{latitude},{longitude}/{p['lat']},{p['lon']}"
                    safezone_lines += f"{i}. {p['name']} ({p['dist']} km)\n   ↳ Navigate: {nav_url}\n"

                # Build a Google Maps search URL that shows all safe zones near location
                # Uses /search/ which pins results on map
                hazard_key = "FIRE"      if any("FIRE"  in h for h in hazard_names) else \
                 "RAINFALL"  if any("RAIN"  in h or "CLOUD" in h for h in hazard_names) else \
                 "LANDSLIDE" if any("LAND"  in h for h in hazard_names) else \
                 "GENERAL"

                maps_search_url = (
        f"https://paraskrishali10.github.io/-disaster-map/"
        f"?lat={latitude}&lon={longitude}&hazard={hazard_key}"
    )

            else:
                maps_search_url = f"https://maps.google.com/?q={latitude},{longitude}"


        except Exception as e:
            maps_search_url = f"https://maps.google.com/?q={latitude},{longitude}"
            safezone_lines = "\n⚠️ Safe zone data unavailable.\n"
            st.warning(f"Safe zone fetch error: {e}")  # ← ADD THIS
        body = (
            f"SOS ALERT\n"
            f"Location: {latitude:.4f},{longitude:.4f}\n"
            f"Threats: {', '.join(a['hazard'] for a in alerts[:3])}\n"
            f"Map: {maps_search_url}"
        )

        message = client.messages.create(
              body=body,
    from_=st.secrets["TWILIO_FROM_NUMBER"],
    to=st.secrets["ALERT_TO_NUMBER"]
        )

        return True, message.sid

    except Exception as e:
        return False, str(e)
# ==========================================
# 1. ENTERPRISE DIRECTORY CONFIGURATION
# ==========================================
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))

FF_MODELS_DIR = r"D:\FINALYEARPROJECT\FORESTFIREPREDICTION\models"
CB_MODELS_DIR = r"D:\FINALYEARPROJECT\CLOUDBURSTHEAVYRAINFALL\models"
LS_MODELS_DIR = r"D:\FINALYEARPROJECT\LANDSLIDEPREDICTOR\models"

# ==========================================
# 2. PAGE CONFIGURATION & CSS
# ==========================================
st.set_page_config(page_title="Multi-Hazard Early Warning System", page_icon="🌍", layout="wide")

st.markdown("""
    <style>
    .big-font { font-size: 18px !important; font-weight: bold; }
    .threat-extreme { background-color: #4a0000; color: white; padding: 20px; border-radius: 8px; text-align: center; border: 2px solid #ff4d4d; margin-bottom: 15px;}
    .threat-high { background-color: #8c2a04; color: white; padding: 20px; border-radius: 8px; text-align: center; border: 2px solid #ff7b00; margin-bottom: 15px;}
    .threat-moderate { background-color: #8c6b04; color: white; padding: 20px; border-radius: 8px; text-align: center; border: 2px solid #ffcc00; margin-bottom: 15px;}
    .threat-low { background-color: #0b4a18; color: white; padding: 20px; border-radius: 8px; text-align: center; border: 2px solid #00ff40; margin-bottom: 15px;}
    .metric-card { background-color: #1e1e1e; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #333;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SYSTEM INITIALIZATION (EARTH ENGINE)
# ==========================================
@st.cache_resource
def init_ee():
    try:
        key_file_path = os.path.join(DASHBOARD_DIR, 'ee-key.json')
        if not os.path.exists(key_file_path):
             return False, "Google Earth Engine Key JSON missing."
        credentials = ee.ServiceAccountCredentials('', key_file_path)
        ee.Initialize(credentials, project='lofty-inn-490212-m9')
        return True, None
    except Exception as e:
        return False, str(e)

ee_ready, ee_error = init_ee()

# ==========================================
# 4. UNIFIED MODEL LOADING ARCHITECTURE
# ==========================================
@st.cache_resource
def load_all_assets():
    assets = {'FF': {}, 'CB': {}, 'LS': {}, 'scaler': None}

    # --- Forest Fire ---
    ff_files = {'Random Forest': 'forest_fire_rf_model.pkl', 'XGBoost': 'forest_fire_xgb_model.pkl', 'MLP Neural Net': 'forest_fire_mlp_model.pkl', 'Hybrid Ensemble': 'forest_fire_hybrid_model.pkl'}
    for name, f in ff_files.items():
        try: assets['FF'][name] = joblib.load(os.path.join(FF_MODELS_DIR, f))
        except: pass

    # --- Cloudburst ---
    cb_files = {'Random Forest': 'model_rf.pkl', 'XGBoost': 'model_xgb.pkl', 'SVM': 'model_svm.pkl', 'Tuned Hybrid': 'model_hybrid_tuned.pkl'}
    for name, f in cb_files.items():
        try: assets['CB'][name] = joblib.load(os.path.join(CB_MODELS_DIR, f))
        except: pass

    # --- Landslide ---
    ls_files = {'Random Forest': 'random_forest_landslide_model.pkl', 'XGBoost': 'xgboost_landslide_model.pkl', 'SVM': 'svm_landslide_model.pkl', 'Hybrid Stacking': 'hybrid_landslide_model.pkl'}
    for name, f in ls_files.items():
        try: assets['LS'][name] = joblib.load(os.path.join(LS_MODELS_DIR, f))
        except: pass

    try: assets['scaler'] = joblib.load(os.path.join(LS_MODELS_DIR, 'scaler.pkl'))
    except Exception as e: st.error(f"Failed to load Landslide Scaler: {e}")

    return assets

# ==========================================
# 5. UNIFIED TELEMETRY EXTRACTION
# ==========================================
def get_unified_gee(lat, lon):
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

def get_unified_weather(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_max,soil_moisture_0_to_7cm_mean,soil_moisture_28_to_100cm_mean&past_days=15&forecast_days=1&timezone=auto"
    try:
        r = requests.get(url).json()['daily']
        return r
    except:
        return None

# ==========================================
# 6. UI: HEADER & SIDEBAR
# ==========================================
st.title("🌍 Integrated Multi-Hazard Early Warning System")
st.markdown("### AI-Driven Risk Assessment for Forest Fires, Cloudbursts, and Landslides in the Himalayan Belt")

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/Uttarakhand_Emblem.png/180px-Uttarakhand_Emblem.png", width=100)
st.sidebar.header("📍 System Target Coordinates")

# latitude = st.sidebar.number_input("Target Latitude", value=30.1459, format="%.4f")
# longitude = st.sidebar.number_input("Target Longitude", value=78.7672, format="%.4f")

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



st.sidebar.markdown("---")

if not ee_ready:
    st.error(f"🛑 Critical System Failure: Google Earth Engine API Offline. Details: {ee_error}")
    st.stop()

assets = load_all_assets()

# ==========================================
# 7. CORE EXECUTION LOGIC
# ==========================================
if st.sidebar.button("INITIATE MULTI-HAZARD ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Synchronizing Satellite & Atmospheric Telemetry..."):

        gee_data = get_unified_gee(latitude, longitude)
        weather_data = get_unified_weather(latitude, longitude)

        if not weather_data:
            st.error("Atmospheric API disconnected. Please try again.")
            st.stop()

        elev, slope, aspect, ndvi = gee_data
        month = datetime.now().month

        temp = weather_data['temperature_2m_max']
        rain = weather_data['precipitation_sum']
        hum = weather_data['relative_humidity_2m_max']
        wind = weather_data['wind_speed_10m_max']
        surf_moisture = weather_data['soil_moisture_0_to_7cm_mean']
        deep_moisture = weather_data['soil_moisture_28_to_100cm_mean']

        max_t_today = temp[15]
        hum_today = hum[15]
        wind_today = wind[15]
        rain_today = rain[15]
        surf_m_today = surf_moisture[15]
        deep_m_today = deep_moisture[15]

        t7_avg = np.mean(temp[8:15])
        r7_sum = np.sum(rain[8:15])
        r3_sum = np.sum(rain[12:15])
        r15_sum = np.sum(rain[0:15])

        # --- FOREST FIRE ---
        fdi = (max_t_today * wind_today) / (hum_today + 1)
        ff_input = pd.DataFrame({
            'Month': [float(month)], 'Rain_7d_Sum': [float(r7_sum)], 'Temp_7d_Avg': [float(t7_avg)],
            'Fire_Danger_Index': [float(fdi)], 'Max_Temperature_C': [float(max_t_today)], 'Max_Humidity_pct': [float(hum_today)],
            'Total_Rainfall_mm': [float(rain_today)], 'Max_Wind_Speed_kmh': [float(wind_today)], 'Elevation_m': [float(elev)],
            'Slope_deg': [float(slope)], 'Aspect_deg': [float(aspect)], 'Baseline_NDVI': [float(ndvi)]
        }).values

        # --- CLOUDBURST ---
        cb_input = pd.DataFrame({
            'Month': [float(month)], 'Rain_7d_Sum': [float(r7_sum)], 'Temp_7d_Avg': [float(t7_avg)],
            'Max_Temperature_C': [float(max_t_today)], 'Max_Humidity_pct': [float(hum_today)], 'Total_Rainfall_mm': [float(rain_today)],
            'Elevation_m': [float(elev)], 'Slope_deg': [float(slope)], 'Aspect_deg': [float(aspect)]
        }).values

        # --- LANDSLIDE ---
        soil_grad = deep_m_today - surf_m_today
        rain_slope = r7_sum * slope
        ls_input = pd.DataFrame({
            'Elevation_m': [elev], 'Slope_deg': [slope], 'Baseline_NDVI': [ndvi],
            'Rainfall_Day_0_mm': [rain_today], 'Rainfall_Antecedent_3D_mm': [r3_sum], 'Rainfall_Antecedent_7D_mm': [r7_sum],
            'Rainfall_Antecedent_15D_mm': [r15_sum], 'Soil_Moisture_Surface': [surf_m_today], 'Soil_Moisture_Deep': [deep_m_today],
            'Month_Sin': [np.sin(2 * np.pi * month / 12)], 'Month_Cos': [np.cos(2 * np.pi * month / 12)],
            'Aspect_Sin': [np.sin(np.radians(aspect))], 'Aspect_Cos': [np.cos(np.radians(aspect))],
            'Soil_Moisture_Gradient': [soil_grad], 'Rain_Slope_Interaction': [rain_slope], 'Total_15D_Water_Load': [r15_sum + deep_m_today],
            'Rain_norm': [rain_today / 100.0], 'Slope_norm': [slope / 90.0], 'Moisture_norm': [surf_m_today],
            'Risk_Score': [(r7_sum * slope) / 100.0], 'random_noise_feature': [np.random.rand()]
        }).values

        if assets['scaler']:
            ls_input_scaled = assets['scaler'].transform(ls_input)
        else:
            ls_input_scaled = None

       # --- PREDICTIONS ---
        pred_ff = assets['FF']['Hybrid Ensemble'].predict(ff_input)[0] if 'Hybrid Ensemble' in assets['FF'] else -1
        pred_cb = assets['CB']['Tuned Hybrid'].predict(cb_input)[0] if 'Tuned Hybrid' in assets['CB'] else -1
        pred_ls = assets['LS']['Hybrid Stacking'].predict(ls_input_scaled)[0] if ls_input_scaled is not None and 'Hybrid Stacking' in assets['LS'] else -1

        # ==========================================
        # 8. MASTER THREAT MATRIX
        # ==========================================
        st.markdown("### 📡 MASTER THREAT MATRIX")
        col1, col2, col3 = st.columns(3)

        with col1:
            if pred_ff == 3: st.markdown('<div class="threat-extreme"><h3>🚨 FOREST FIRE</h3><p>EXTREME IGNITION PROBABILITY</p></div>', unsafe_allow_html=True)
            elif pred_ff == 2: st.markdown('<div class="threat-high"><h3>⚠️ FOREST FIRE</h3><p>HIGH FIRE DANGER CONDITIONS</p></div>', unsafe_allow_html=True)
            elif pred_ff == 1: st.markdown('<div class="threat-moderate"><h3>🔔 FOREST FIRE</h3><p>MODERATE IGNITION RISK</p></div>', unsafe_allow_html=True)
            else: st.markdown('<div class="threat-low"><h3>✅ FOREST FIRE</h3><p>STABLE CONDITIONS</p></div>', unsafe_allow_html=True)

        with col2:
            if pred_cb == 2: st.markdown('<div class="threat-extreme"><h3>🚨 CLOUDBURST</h3><p>CRITICAL ATMOSPHERIC INSTABILITY</p></div>', unsafe_allow_html=True)
            elif pred_cb == 1: st.markdown('<div class="threat-high"><h3>⚠️ HEAVY RAINFALL</h3><p>ELEVATED PRECIPITATION RISK</p></div>', unsafe_allow_html=True)
            else: st.markdown('<div class="threat-low"><h3>✅ PRECIPITATION</h3><p>NORMAL WEATHER PATTERNS</p></div>', unsafe_allow_html=True)

        with col3:
            if pred_ls == 2: st.markdown('<div class="threat-extreme"><h3>🚨 LANDSLIDE</h3><p>CRITICAL SLOPE INSTABILITY</p></div>', unsafe_allow_html=True)
            elif pred_ls == 1: st.markdown('<div class="threat-high"><h3>⚠️ LANDSLIDE</h3><p>ELEVATED MOISTURE & STRAIN</p></div>', unsafe_allow_html=True)
            else: st.markdown('<div class="threat-low"><h3>✅ TERRAIN</h3><p>GEOLOGICALLY STABLE</p></div>', unsafe_allow_html=True)

        # ==========================================
        # SOS ALERT LOGIC
        # ==========================================
       # ==========================================
        # SOS ALERT LOGIC — checks Hybrid + any base model
        # ==========================================
        active_alerts = []

        # --- Forest Fire: Hybrid primary, fallback to any base model ---
        ff_map_rev = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Extreme'}
        ff_base_preds = {name: model.predict(ff_input)[0] for name, model in assets['FF'].items()}
        ff_any_high = any(v >= 2 for v in ff_base_preds.values())  # any model says High/Extreme

        if pred_ff == 3:
            active_alerts.append({"hazard": "FOREST FIRE", "level": "EXTREME",
                                   "detail": f"Hybrid: Extreme | Base models: {', '.join(f'{k}={ff_map_rev[v]}' for k,v in ff_base_preds.items())}"})
        elif pred_ff == 2:
            active_alerts.append({"hazard": "FOREST FIRE", "level": "HIGH",
                                   "detail": f"Hybrid: High | Base models: {', '.join(f'{k}={ff_map_rev[v]}' for k,v in ff_base_preds.items())}"})
        elif ff_any_high:
            triggering = [k for k, v in ff_base_preds.items() if v >= 2]
            active_alerts.append({"hazard": "FOREST FIRE", "level": "HIGH (BASE MODEL)",
                                   "detail": f"Triggered by: {', '.join(triggering)} | Hybrid: Low/Moderate"})

        # --- Cloudburst: Hybrid primary, fallback to any base model ---
        cb_map_rev = {0: 'Normal', 1: 'Heavy Rain', 2: 'Cloudburst'}
        cb_base_preds = {name: model.predict(cb_input)[0] for name, model in assets['CB'].items()}
        cb_any_high = any(v >= 1 for v in cb_base_preds.values())  # any model says Heavy Rain or worse

        if pred_cb == 2:
            active_alerts.append({"hazard": "CLOUDBURST", "level": "CRITICAL",
                                   "detail": f"Hybrid: Cloudburst | Base models: {', '.join(f'{k}={cb_map_rev[v]}' for k,v in cb_base_preds.items())}"})
        elif pred_cb == 1:
            active_alerts.append({"hazard": "HEAVY RAINFALL", "level": "ELEVATED",
                                   "detail": f"Hybrid: Heavy Rain | Base models: {', '.join(f'{k}={cb_map_rev[v]}' for k,v in cb_base_preds.items())}"})
        elif cb_any_high:
            triggering = [k for k, v in cb_base_preds.items() if v >= 1]
            active_alerts.append({"hazard": "HEAVY RAINFALL", "level": "ELEVATED (BASE MODEL)",
                                   "detail": f"Triggered by: {', '.join(triggering)} | Hybrid: Normal"})

        # --- Landslide: Hybrid primary, fallback to any base model ---
        ls_map_rev = {0: 'Stable', 1: 'Moderate Risk', 2: 'High Risk'}
        if ls_input_scaled is not None:
            ls_base_preds = {name: model.predict(ls_input_scaled)[0] for name, model in assets['LS'].items()}
            ls_any_high = any(v >= 1 for v in ls_base_preds.values())
        else:
            ls_base_preds = {}
            ls_any_high = False

        if pred_ls == 2:
            active_alerts.append({"hazard": "LANDSLIDE", "level": "CRITICAL",
                                   "detail": f"Hybrid: High Risk | Base models: {', '.join(f'{k}={ls_map_rev[v]}' for k,v in ls_base_preds.items())}"})
        elif pred_ls == 1:
            active_alerts.append({"hazard": "LANDSLIDE", "level": "ELEVATED",
                                   "detail": f"Hybrid: Moderate Risk | Base models: {', '.join(f'{k}={ls_map_rev[v]}' for k,v in ls_base_preds.items())}"})
        elif ls_any_high:
            triggering = [k for k, v in ls_base_preds.items() if v >= 1]
            active_alerts.append({"hazard": "LANDSLIDE", "level": "ELEVATED (BASE MODEL)",
                                   "detail": f"Triggered by: {', '.join(triggering)} | Hybrid: Stable"})

        st.markdown("---")

        if active_alerts:
            st.error(f"🚨 {len(active_alerts)} ACTIVE THREAT(S) DETECTED — SOS ALERT TRIGGERED")
            with st.spinner("📡 Transmitting SOS Alert via Twilio..."):
                success, result = send_sos_alert(latitude, longitude, active_alerts)
            if success:
                st.success(f"✅ SOS SMS sent! Message SID: `{result}`")
                try:
                    debug_client = Client(st.secrets["TWILIO_ACCOUNT_SID"], st.secrets["TWILIO_AUTH_TOKEN"])
                    msg = debug_client.messages(result).fetch()
                    st.info(f"**Twilio Status:** `{msg.status}` | **Error:** `{msg.error_message}`")
                except Exception as debug_err:
                    st.warning(f"Could not fetch status: {debug_err}")
            else:
                st.warning(f"⚠️ Alert transmission failed: {result}")
        else:
            st.success("✅ No critical threats detected. No SOS alert required.")
        # ==========================================
        # 9. RAW TELEMETRY DATA EXPANDER (EXPANDED)
        # ==========================================
        with st.expander("🔬 View Extracted Geospatial & Meteorological Telemetry"):
            st.markdown("#### ⛰️ Topographical and Meteorological Data")
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Elevation", f"{elev:.0f} m")
            t2.metric("Slope Angle", f"{slope:.1f}°")
            t3.metric("Aspect (Facing)", f"{aspect:.1f}°")
            t4.metric("Vegetation (NDVI)", f"{ndvi:.3f}")

            st.markdown("#### 🌤️ Current Atmospheric Conditions")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Max Temperature", f"{max_t_today:.1f}°C")
            a2.metric("Relative Humidity", f"{hum_today:.0f}%")
            a3.metric("Max Wind Speed", f"{wind_today:.1f} km/h")
            a4.metric("Fire Danger Index", f"{fdi:.2f}")

            st.markdown("#### 💧 Hydrological & Soil Profile")
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("7-Day Rainfall Accum.", f"{r7_sum:.1f} mm")
            h2.metric("15-Day Rainfall Accum.", f"{r15_sum:.1f} mm")
            h3.metric("Surface Moisture (0-7cm)", f"{surf_m_today:.3f} m³/m³")
            h4.metric("Deep Moisture (28-100cm)", f"{deep_m_today:.3f} m³/m³")

        # ==========================================
        # 10. MODEL SUMMARY
        # ==========================================
        st.markdown("### 🧠 Model Summary")
        tab1, tab2, tab3 = st.tabs(["🌲 Fire Analytics", "⛈️ Precipitation Analytics", "⛰️ Terrain Analytics"])

        with tab1:
            st.write("Predictions from individual base models for Forest Fire Ignition.")
            c1 = st.columns(len(assets['FF']))
            ff_map = {0: 'Low', 1: 'Moderate', 2: 'High', 3: 'Extreme'}
            for i, (name, model) in enumerate(assets['FF'].items()):
                res = model.predict(ff_input)[0]
                c1[i].info(f"**{name}**\n\nResult: {ff_map[res]}")

        with tab2:
            st.write("Predictions from individual base models for Extreme Rainfall.")
            c2 = st.columns(len(assets['CB']))
            cb_map = {0: 'Normal Weather', 1: 'Heavy Rain', 2: 'Cloudburst Warning'}
            for i, (name, model) in enumerate(assets['CB'].items()):
                res = model.predict(cb_input)[0]
                c2[i].info(f"**{name}**\n\nResult: {cb_map[res]}")

        with tab3:
            st.write("Predictions from individual models for Landslide Risk.")
            c3 = st.columns(len(assets['LS']))
            ls_map = {0: 'Stable', 1: 'Moderate Risk', 2: 'High Risk'}
            for i, (name, model) in enumerate(assets['LS'].items()):
                if ls_input_scaled is not None:
                    res = model.predict(ls_input_scaled)[0]
                    c3[i].info(f"**{name}**\n\nResult: {ls_map[res]}")
                else:
                    c3[i].error("Scaler Offline")