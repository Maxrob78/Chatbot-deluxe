@echo off
cd /d "%~dp0"

echo ============================================
echo  Chatbot 2000 Pro
echo ============================================
echo.

:: Vérifier que Python est installé
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] Python n'est pas installe ou pas dans le PATH.
    echo Telechargez Python sur https://www.python.org/downloads/
    echo Cochez bien "Add Python to PATH" lors de l'installation.
    pause
    exit /b 1
)

:: Vérifier et installer les dépendances à chaque lancement (très rapide si déjà installées)
echo Verification des dependances...
python -m pip install -r requirements.txt >nul
if errorlevel 1 (
    echo [ERREUR] L'installation des dependances a echoue.
    echo Essayez manuellement : pip install -r requirements.txt
    pause
    exit /b 1
)
echo Dependances OK.
echo.

:: Tuer l'ancien serveur si actif
echo Arret de l'ancien serveur si actif...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000 "') do (
    taskkill /F /PID %%a >nul 2>&1
)
timeout /t 2 /nobreak >nul

:: Lancer
echo Demarrage du serveur...
start "" http://localhost:8000

echo.
echo  Chatbot 2000 tourne sur http://localhost:8000
echo  Fermez cette fenetre pour arreter le serveur.
echo.

python -m uvicorn main:app --port 8000
pause