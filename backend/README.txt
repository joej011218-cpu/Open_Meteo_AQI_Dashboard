AUTO-REFRESH BACKEND UPDATE

Replace these files in your existing Open_Meteo/backend folder:
- app.py
- database.py
- service.py

What changed:
1. /api/latest and /api/forecast now automatically refresh when:
   - SQLite is empty after a Render restart/redeploy,
   - no forecast exists, or
   - the last successful refresh is at least 60 minutes old.

2. A Lock prevents the frontend's simultaneous /api/latest and /api/forecast
   requests from launching duplicate Open-Meteo/XGBoost refreshes.

3. SQLite app_state records:
   - last_refresh_at
   - last_refresh_error

4. If Open-Meteo temporarily fails but cached data exists, the backend serves
   cached data instead of taking the dashboard offline.

5. GET / now returns a friendly API description.

DEPLOY:
cd ~/Open_Meteo
git add backend/app.py backend/database.py backend/service.py
git commit -m "Add automatic Open-Meteo refresh on Render"
git push

After Render redeploys:
https://YOUR-SERVICE.onrender.com/api/health

You no longer need to manually POST /api/refresh after a normal Render restart.
The first /api/latest or /api/forecast request will rebuild the data automatically.
