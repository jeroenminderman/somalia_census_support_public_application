REM The tool expects the user to be in the src/ folder so we navigate there
cd src

REM Using the virtual environment named "venv-somalia-app" to run the streamlit application
"../venv-somalia-app/Scripts/python.exe" "../venv-somalia-app/Scripts/streamlit.exe" run homepage.py --server.port=8081