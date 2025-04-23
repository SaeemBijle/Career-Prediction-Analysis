# 🧠 Career Prediction Analysis

**Career Prediction Analysis** is a machine learning-powered web and desktop application that predicts the most suitable career path for students based on their academic marks and extracurricular skills. The model considers technical skills, certifications, and soft skills like public speaking to offer tailored career role suggestions.

---

## 🚀 Features

- 🎯 Predicts job roles from over **30+ career domains**
- 📚 Uses subject-wise academic performance (Sem 1 to Sem 8)
- 💼 Includes co-curricular and soft skill inputs:
  - Hackathon participation
  - Competitive coding experience
  - Public speaking ability
  - Certifications
- 🤖 Trained ML model using Random Forest Classifier
- 🌐 Web interface using Django, HTML & CSS

---

## 📥 Inputs Considered

- **Subject Marks Like (OS,AOA,PC....etc)**
- **Number Of Hackathon Participation**
- **Coding Skill Level** (1-10)
- **Public Speaking Skill** (1-10)
- **Certifications**
-**And Others..... (Refer Dataset)**



## 🛠️ Tech Stack

| Layer        | Technology            |
|--------------|------------------------|
| Frontend     | HTML, CSS              |
| Backend      | Django Flask (Python)        |
| ML & Data    | scikit-learn, pandas   |
| Model        | RandomForestClassifier |         
| Utilities    | joblib, seaborn, matplotlib |



## 🧪 How It Works

1. **User Input:** The user enters marks and other skill-related data.
2. **Feature Engineering:** The data is processed and encoded into vectors.
3. **Model Prediction:** Trained ML model predicts the best career domain.
4. **Output:** Predicted job role is displayed with recommendations.



## 📈 Example Output

 **Predicted Role:** Programmer Analyst

 -**Description:** Designs and implements technology solutions to meet business needs, focusing on
 scalability, security, and performance.
 -**Skills**: System architecture, cloud computing, solution design, business analysis, project
 management.
 -**Potential Companies**: Amazon, Microsoft, Google, consulting firms, large enterprises



career-prediction-analysis/
├── static/                          # 🎨 Static files: CSS and other assets
│   └── style.css                    # CSS file for styling HTML layout and design
│
├── templates/                       # 📄 HTML templates for frontend
│   └── index.html                   # Main form page for user input and displaying output
│
├── dataset.xlsx                     # 📊 Raw dataset used for training the ML model
│
├── ml/                              # 📦 Folder for Jupyter notebooks and related files
│   └── ml.ipynb                     # Jupyter Notebook for ML model training and preprocessing
│
├── model/                           # 🧠 Folder for model and preprocessing files
│   ├── columns.pkl                  # 🔑 Column transformer: stores feature encoding and preprocessing steps
│   ├── scaler.pkl                   # 🔧 Scaler: stores scaling information (e.g., MinMaxScaler, StandardScaler)
│   └── le.pkl                       # 🧮 Label Encoder: stores encoding for categorical variables
│ 
│   ➡️ **Note:** These `.pkl` files will be auto-generated after running the ML model in `ml.ipynb`
│
└── app.py                           # ⚙️ Backend logic for the app, handles user input and prediction process


