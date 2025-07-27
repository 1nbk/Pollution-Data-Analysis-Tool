# 🌍 Numerical Analysis of Pollution Data with Streamlit

[![Streamlit App](https://img.shields.io/badge/Launch-App-green?logo=streamlit)](https://pollution-data-analysis-tool-atb7jleirzfsotg7ap4kad.streamlit.app/)


An interactive Python application for pollution data exploration and environmental analysis, combining advanced numerical methods with an intuitive [Streamlit](https://streamlit.io/) interface.

---

## 🔍 Overview

This project provides a clean separation between:
- 🧮 **Numerical Computation** (`pollution_analyzer_logic.py`)
- 🖥️ **Interactive UI** (`streamlit_app.py`)

It analyzes pollution concentration data using a high-accuracy **Romberg Integrator** and enables interactive visualizations, threshold-based assessments, and downloadable reports.

---

## 🚀 Live Demo

👉 **Try it here**:  
🔗 [Pollution Data Analysis Tool (Streamlit App)](https://pollution-data-analysis-tool-atb7jleirzfsotg7ap4kad.streamlit.app/)

---

## 📊 Features

- ✅ **Romberg Integration** for precise area-under-curve estimation
- 📈 **Visualization** of pollution trends with Matplotlib
- 🧠 **Peak Exposure Analysis**: Find time periods exceeding safe pollution thresholds
- 📅 **Daily Average Calculation**
- 📄 **Report Generation** with total exposure and health risk summaries
- 🧪 **Sample CSV Generator** for quick testing
- 🔁 Upload custom CSVs for real-world pollution data
- 📥 Downloadable output and CSV data

---

## 🧠 Technical Highlights

### 🧮 RombergIntegrator (Numerical Engine)

- **Tolerance:** `1e-6` for convergence accuracy
- **Max Iterations:** `20` to prevent infinite loops
- **Richardson Extrapolation** over the Trapezoidal Rule

### 🧹 PollutionDataAnalyzer

- Generate synthetic data
- Load and parse CSVs
- Interpolate data for continuous analysis
- Calculate total exposure, peak periods, and daily averages

---

## 🧑‍💻 Tech Stack

| Component             | Tech Used                     |
|----------------------|-------------------------------|
| 🐍 Language           | Python 3                      |
| 📊 Data Analysis      | NumPy, SciPy, Pandas          |
| 🎨 Visualization      | Matplotlib                    |
| 🖥️ Web UI             | Streamlit                     |

---

## 🗂️ Project Structure

```
📁 Numerical-Analysis-Project/
├── pollution_analyzer_logic.py      # Numerical integration & data analysis
├── streamlit_app.py                 # Streamlit frontend
├── pollution_data.csv               # Sample dataset (generated)
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

---

## 📥 Installation

```bash
# Clone the repo
git clone https://github.com/1nbk/Pollution-Data-Analysis-Tool.git
cd Pollution-Data-Analysis-Tool

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run streamlit_app.py
```

---

## 📸 Screenshots

> You can include these once you take screenshots of the deployed app interface and analysis visuals.

---

## 📄 License

This project is for academic purposes. Feel free to fork and adapt for learning or environmental research use cases. Contact the me  for extended use.
https://1nbk.github.io/myPortfolio/ 

---

## 🙏 Acknowledgements

Special thanks to the course instructors and [Streamlit](https://streamlit.io/) for making rapid data apps easy to build!
