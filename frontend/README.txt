Open_Meteo dashboard files

Copy both files into your existing Open_Meteo/frontend folder:

- prediction.html
- precautions.html

Backend expected at:
http://127.0.0.1:5000

Required endpoints:
- GET /api/latest
- GET /api/forecast

Before opening the dashboard:
1. Start backend: python app.py
2. Ensure data/forecast exists: python refresh_once.py
3. From frontend folder: python -m http.server 5500
4. Open: http://127.0.0.1:5500/prediction.html

This version preserves the supplied dark/light dashboard design, forecast cards,
table, chart, AQI colors, health-precaution modal, and precautions page.
It removes NH3 because the Open-Meteo Vadodara runtime pipeline uses:
PM2.5, PM10, NO2, SO2, CO, O3.
