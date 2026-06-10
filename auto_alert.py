import requests
import numpy as np
import joblib
import os
from datetime import datetime
from twilio.rest import Client

# ==========================================
# CONFIGURATION
# ==========================================
TWILIO_SID       = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_TOKEN     = os.environ["TWILIO_AUTH_TOKEN"]
ALERT_TO         = os.environ["ALERT_TO_NUMBER"]
GEOAPIFY_KEY     = os.environ["GEOAPIFY_API_KEY"]

# Target locations to monitor (add as many as needed)
MONITOR_LOCATIONS = [
    {"name": "Pauri Garhwal",   "lat": 30.1840, "lon": 78.6906},
    {"name": "Rishikesh",       "lat": 30.0869, "lon": 78.2676},
    {"name": "Chamoli",         "lat": 30.4087, "lon": 79.3212},
    {"name": "Uttarkashi",      "lat": 30.7268, "lon": 78.4354},
    {"name": "Pithoragarh",     "lat": 29.5830, "lon": 80.2180},
]

FF_MODELS_DIR = "models/ff"
CB_MODELS_DIR = "models/cb"
LS_MODELS_DIR = "models/ls"

# ==========================================
# WEATHER DATA
# ==========================================
def get_weather(lat, lon):
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&daily=temperature_2m_max,precipitation_sum,wind_speed_10m_max,"
        f"relative_humidity_2m_max,soil_moisture_0_to_7cm_mean,"
        f"soil_moisture_28_to_100cm_mean"
        f"&past_days=15&forecast_days=1&timezone=auto"
    )
    try:
        return requests.get(url, timeout=10).json()["daily"]
    except:
        return None

# ==========================================
# TERRAIN DATA (static fallback — no EE needed)
# ==========================================
TERRAIN_DATA = {
    "Pauri Garhwal": {"elev": 1814, "slope": 28, "aspect": 180, "ndvi": 0.42},
    "Rishikesh":     {"elev": 356,  "slope": 8,  "aspect": 135, "ndvi": 0.38},
    "Chamoli":       {"elev": 2580, "slope": 32, "aspect": 200, "ndvi": 0.35},
    "Uttarkashi":    {"elev": 1158, "slope": 25, "aspect": 160, "ndvi": 0.40},
    "Pithoragarh":   {"elev": 1814, "slope": 30, "aspect": 175, "ndvi": 0.36},
}

# ==========================================
# LOAD MODELS
# ==========================================
def load_models():
    assets = {"FF": {}, "CB": {}, "LS": {}, "scaler": None}
    ff_files = {
        "Hybrid Ensemble": f"{FF_MODELS_DIR}/forest_fire_hybrid_model.pkl"
    }
    cb_files = {
        "Tuned Hybrid": f"{CB_MODELS_DIR}/model_hybrid_tuned.pkl"
    }
    ls_files = {
        "Hybrid Stacking": f"{LS_MODELS_DIR}/hybrid_landslide_model.pkl"
    }
    for name, f in ff_files.items():
        try: assets["FF"][name] = joblib.load(f)
        except: pass
    for name, f in cb_files.items():
        try: assets["CB"][name] = joblib.load(f)
        except: pass
    for name, f in ls_files.items():
        try: assets["LS"][name] = joblib.load(f)
        except: pass
    try: assets["scaler"] = joblib.load(f"{LS_MODELS_DIR}/scaler.pkl")
    except: pass
    return assets

# ==========================================
# RUN PREDICTION FOR ONE LOCATION
# ==========================================
def predict_location(loc, assets):
    weather = get_weather(loc["lat"], loc["lon"])
    if not weather:
        return []

    terrain = TERRAIN_DATA.get(loc["name"],
        {"elev": 1500, "slope": 15, "aspect": 180, "ndvi": 0.35})

    elev   = terrain["elev"]
    slope  = terrain["slope"]
    aspect = terrain["aspect"]
    ndvi   = terrain["ndvi"]
    month  = datetime.now().month

    temp         = weather["temperature_2m_max"]
    rain         = weather["precipitation_sum"]
    hum          = weather["relative_humidity_2m_max"]
    wind         = weather["wind_speed_10m_max"]
    surf_moisture = weather["soil_moisture_0_to_7cm_mean"]
    deep_moisture = weather["soil_moisture_28_to_100cm_mean"]

    max_t   = temp[15];  hum_t  = hum[15]
    wind_t  = wind[15];  rain_t = rain[15]
    surf_m  = surf_moisture[15]
    deep_m  = deep_moisture[15]

    t7_avg  = np.mean(temp[8:15])
    r7_sum  = np.sum(rain[8:15])
    r3_sum  = np.sum(rain[12:15])
    r15_sum = np.sum(rain[0:15])
    fdi     = (max_t * wind_t) / (hum_t + 1)

    import pandas as pd

    ff_input = pd.DataFrame({
        "Month": [float(month)], "Rain_7d_Sum": [float(r7_sum)],
        "Temp_7d_Avg": [float(t7_avg)], "Fire_Danger_Index": [float(fdi)],
        "Max_Temperature_C": [float(max_t)], "Max_Humidity_pct": [float(hum_t)],
        "Total_Rainfall_mm": [float(rain_t)], "Max_Wind_Speed_kmh": [float(wind_t)],
        "Elevation_m": [float(elev)], "Slope_deg": [float(slope)],
        "Aspect_deg": [float(aspect)], "Baseline_NDVI": [float(ndvi)]
    }).values

    cb_input = pd.DataFrame({
        "Month": [float(month)], "Rain_7d_Sum": [float(r7_sum)],
        "Temp_7d_Avg": [float(t7_avg)], "Max_Temperature_C": [float(max_t)],
        "Max_Humidity_pct": [float(hum_t)], "Total_Rainfall_mm": [float(rain_t)],
        "Elevation_m": [float(elev)], "Slope_deg": [float(slope)],
        "Aspect_deg": [float(aspect)]
    }).values

    soil_grad  = deep_m - surf_m
    rain_slope = r7_sum * slope

    ls_raw = pd.DataFrame({
        "Elevation_m": [elev], "Slope_deg": [slope],
        "Baseline_NDVI": [ndvi], "Rainfall_Day_0_mm": [rain_t],
        "Rainfall_Antecedent_3D_mm": [r3_sum],
        "Rainfall_Antecedent_7D_mm": [r7_sum],
        "Rainfall_Antecedent_15D_mm": [r15_sum],
        "Soil_Moisture_Surface": [surf_m], "Soil_Moisture_Deep": [deep_m],
        "Month_Sin": [np.sin(2 * np.pi * month / 12)],
        "Month_Cos": [np.cos(2 * np.pi * month / 12)],
        "Aspect_Sin": [np.sin(np.radians(aspect))],
        "Aspect_Cos": [np.cos(np.radians(aspect))],
        "Soil_Moisture_Gradient": [soil_grad],
        "Rain_Slope_Interaction": [rain_slope],
        "Total_15D_Water_Load": [r15_sum + deep_m],
        "Rain_norm": [rain_t / 100.0], "Slope_norm": [slope / 90.0],
        "Moisture_norm": [surf_m],
        "Risk_Score": [(r7_sum * slope) / 100.0],
        "random_noise_feature": [np.random.rand()]
    }).values

    ls_input = assets["scaler"].transform(ls_raw) \
               if assets["scaler"] is not None else None

    alerts = []

    if "Hybrid Ensemble" in assets["FF"]:
        pred = assets["FF"]["Hybrid Ensemble"].predict(ff_input)[0]
        if pred >= 2:
            alerts.append({
                "hazard": "FOREST FIRE",
                "level": "EXTREME" if pred == 3 else "HIGH",
                "detail": f"FDI={fdi:.1f} Temp={max_t:.1f}C Wind={wind_t:.1f}kmh"
            })

    if "Tuned Hybrid" in assets["CB"]:
        pred = assets["CB"]["Tuned Hybrid"].predict(cb_input)[0]
        if pred >= 1:
            alerts.append({
                "hazard": "CLOUDBURST" if pred == 2 else "HEAVY RAINFALL",
                "level": "CRITICAL" if pred == 2 else "ELEVATED",
                "detail": f"Rain7d={r7_sum:.1f}mm Humidity={hum_t:.0f}%"
            })

    if ls_input is not None and "Hybrid Stacking" in assets["LS"]:
        pred = assets["LS"]["Hybrid Stacking"].predict(ls_input)[0]
        if pred >= 1:
            alerts.append({
                "hazard": "LANDSLIDE",
                "level": "CRITICAL" if pred == 2 else "ELEVATED",
                "detail": f"Slope={slope}deg Moisture={surf_m:.3f}"
            })

    return alerts

# ==========================================
# SEND WHATSAPP ALERT
# ==========================================
def send_alert(loc, alerts):
    client = Client(TWILIO_SID, TWILIO_TOKEN)

    hazard_names = [a["hazard"] for a in alerts]
    hazard_key = "FIRE"      if any("FIRE"  in h for h in hazard_names) else \
                 "RAINFALL"  if any("RAIN"  in h or "CLOUD" in h
                                    for h in hazard_names) else \
                 "LANDSLIDE" if any("LAND"  in h for h in hazard_names) else \
                 "GENERAL"

    map_url = (
        f"https://paraskrishali10.github.io/-disaster-map/"
        f"?lat={loc['lat']}&lon={loc['lon']}&hazard={hazard_key}"
    )

    threat_lines = "\n".join(
        [f"🔴 {a['hazard']}: {a['level']}" for a in alerts]
    )

    body = (
        f"🚨 AUTO HAZARD ALERT\n"
        f"📍 {loc['name']}\n"
        f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"⚠️ THREATS:\n{threat_lines}\n\n"
        f"🗺️ Safe Zones: {map_url}"
    )

    client.messages.create(
        body=body,
        from_="whatsapp:+14155238886",
        to=f"whatsapp:{ALERT_TO}"
    )
    print(f"✅ Alert sent for {loc['name']}")

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"🔍 Running hazard scan at {datetime.now()}")
    assets = load_models()

    for loc in MONITOR_LOCATIONS:
        print(f"  Checking {loc['name']}...")
        alerts = predict_location(loc, assets)
        if alerts:
            print(f"  ⚠️  {len(alerts)} threats found — sending alert")
            send_alert(loc, alerts)
        else:
            print(f"  ✅  No threats")

    print("✅ Scan complete")