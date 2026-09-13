@echo off
echo ================================================================
echo   TraceToxicAI - Generating Public Link for Remote Access
echo ================================================================
echo.
echo Connecting Cloudflare Secure Tunnel to http://localhost:8501...
echo Look for the link ending in ".trycloudflare.com" below.
echo Share that https://... link with anyone 60 km away!
echo.
echo ================================================================
cd /d "%~dp0"
.\cloudflared.exe tunnel --url http://localhost:8501
pause
