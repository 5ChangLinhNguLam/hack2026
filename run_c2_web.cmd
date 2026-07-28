@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo C2 virtual environment not found: %CD%\.venv
    echo Create it with: py -3.12 -m venv .venv
    pause
    exit /b 1
)

echo Starting C2 Streamlit web from %CD%\.venv
echo Open http://localhost:8501 if the browser does not open automatically.
".venv\Scripts\python.exe" -m streamlit run c2\streamlit_app.py

endlocal
