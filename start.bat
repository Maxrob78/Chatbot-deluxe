@echo off
setlocal enabledelayedexpansion
title Chatbot Deluxe 2000 - Professional Agent
color 0B
cd /d "%~dp0"

echo ========================================================
echo.          AGENTIC DEVELOPMENT FLOW
echo ========================================================
echo [SYSTEM] Verification de l'environnement...

:: 1. Verifier Python
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    echo Telechargez Python sur https://www.python.org/downloads/
    echo Cochez bien "Add Python to PATH" lors de l'installation.
    pause
    exit /b 1
)

:: 2. Verifier les dependances
echo [SYSTEM] Verification des dependances (pip install)...
python -m pip install -r requirements.txt >nul
if errorlevel 1 (
    color 0C
    echo [ERREUR] L'installation des dependances a echoue.
    echo Essayez manuellement : pip install -r requirements.txt
    pause
    exit /b 1
)
echo [OK]     Dependances verifiees.

:: 3. Verifier la config (cle API)
if exist "config.json" (
    findstr /C:"\"api_key\": \"\"" config.json >nul
    if not errorlevel 1 (
        color 0E
        echo [WARNING] Votre cle API OpenRouter semble vide dans config.json.
        echo           Saisissez-la dans les parametres du Chatbot une fois ouvert.
    ) else (
        echo [OK]     Configuration trouvee.
    )
) else (
    echo [INFO]    Premier lancement - config.json sera cree par l'application.
)

:: 4. Nettoyage
echo [SYSTEM] Arret de l'ancienne instance sur le port 8000...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 1 /nobreak >nul

:: 5. Lancement
echo [SYSTEM] Demarrage du serveur Uvicorn...
start "" http://localhost:8000

echo.
echo ========================================================
echo  URL ACCESSIBLE : http://localhost:8000
echo  LOGS DU SERVEUR : (voir ci-dessous)
echo ========================================================
echo.

python -m uvicorn main:app --port 8000 --log-level info
pause