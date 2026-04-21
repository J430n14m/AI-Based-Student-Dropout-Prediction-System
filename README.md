🎓 Student Dropout Prediction & Counseling SystemThis project is an AI-driven tool designed to help educational institutions identify students at risk of dropping out. Using a Random Forest Classifier, the system analyzes academic and financial data to provide real-time risk assessments and actionable counseling advice.📁 Folder Structure & File PlacementTo ensure the system runs correctly, save your files exactly in this hierarchy:PlaintextStudent_Dropout_System/
│
├── dataset/
│   └── student_data.csv          # The raw historical student data
│
├── backend/
│   ├── model.py                  # Script to train the AI
│   ├── app.py                    # The Flask server (API)
│   ├── model.pkl                 # Generated AI "Brain" (after running model.py)
│   └── scaler.pkl                # Generated Data Scaler (after running model.py)
│
└── frontend/
    ├── index.html                # The user interface (Dashboard)
    └── style.css                 # The visual design of the dashboard
🛠️ Setup and Installation
1. Install PythonDownload and install Python (v3.10 or higher) from python.org.IMPORTANT: Check the box "Add Python to PATH" during installation.
2. Install Required LibrariesOpen your terminal (Command Prompt or VS Code Terminal) and run:Bashpip install flask flask-cors scikit-learn pandas numpy
