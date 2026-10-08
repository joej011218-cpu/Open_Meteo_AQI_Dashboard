Open-Meteo full dashboard integration

This package connects:
- Live page -> /api/latest
- Analytics -> /api/history
- Alerts -> /api/alerts
- Prediction -> existing /api/forecast page

BACKEND
Replace these files in your Open_Meteo/backend folder:
- app.py
- database.py
- service.py

The backend now:
1. Fetches Open-Meteo CAMS hourly pollutant values.
2. Stores PM2.5, PM10, NO2, SO2, CO and O3 in SQLite.
3. Calculates CPCB-style AQI using your existing aqi.py.
4. Stores each calculated hourly AQI back into the air_quality table.
5. Exposes /api/history for Analytics.
6. Exposes /api/alerts for stored AQI >= 201.
7. Auto-refreshes at most every 30 minutes when a dashboard API is requested.

DASHBOARD
Replace:
- dashboard/analytics.html
- dashboard/alerts.html

The Live page can use the separately supplied Open-Meteo-connected index.html.

DEPLOY BACKEND
cd ~/Open_Meteo
git add backend/app.py backend/database.py backend/service.py
git commit -m "Connect analytics and alerts to stored Open Meteo AQI"
git push

After Render backend is Live, test:
https://open-meteo-aqi-dashboard.onrender.com/api/history?hours=24
https://open-meteo-aqi-dashboard.onrender.com/api/alerts?limit=200

DEPLOY PUBLIC DASHBOARD
Copy analytics.html and alerts.html to:
~/Documents/AIoT_Public_Dashboard/dashboard/

Then:
cd ~/Documents/AIoT_Public_Dashboard
git add dashboard/analytics.html dashboard/alerts.html
git commit -m "Connect analytics and alerts to Open Meteo backend"
git push origin main
