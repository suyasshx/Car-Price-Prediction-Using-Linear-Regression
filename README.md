# 🚗 Car Price Prediction using Linear Regression

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Model-Linear%20Regression-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow)

This project implements a **Linear Regression–based machine learning model** to predict the selling price of used cars.  
It includes **data analysis, visualization, model training, evaluation**, and a **Tkinter-based desktop GUI** for real-time car price prediction.

---

## 📌 Features

- Complete machine learning pipeline  
- Exploratory Data Analysis (EDA) with Matplotlib and Seaborn  
- Feature encoding and scaling  
- Linear Regression model training and evaluation  
- Performance metrics: MAE, MSE, R² Score  
- Interactive Tkinter GUI for live price prediction  

---

## 📊 Tech Stack

- **Language:** Python  
- **Libraries:**  
  - pandas  
  - matplotlib  
  - seaborn  
  - scikit-learn  
  - tkinter  
- **Model:** Linear Regression  

---

## 📁 Project Structure

```text
Linear-Regression-Model/
│
├── car_data.csv          # Dataset used for training and testing
├── car_pred_model.py    # Data preprocessing, EDA, model training & evaluation
├── tinker.py            # Tkinter-based GUI for price prediction
└── README.md            # Project documentation
```

---

## 🔍 Project Workflow

### 1️⃣ Data Loading & Exploration
- Loaded dataset using pandas  
- Checked for null values and data distribution  
- Analyzed categorical variables:
  - Fuel Type  
  - Seller Type  
  - Transmission  

### 2️⃣ Data Visualization
- Bar plots for categorical features vs selling price  
- Seaborn bar plots for better comparison  
- Correlation heatmap to understand feature relationships  
- Regression plots for price prediction trends  

### 3️⃣ Data Preprocessing
- Encoded categorical variables  
- Applied one-hot encoding  
- Scaled numerical features using `StandardScaler`  
- Created car age feature from year  

### 4️⃣ Model Training & Evaluation
- Split data into training and testing sets  
- Trained a Linear Regression model  
- Evaluated model using:
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - R² Score  
- Visualized actual vs predicted prices  

### 5️⃣ GUI Application
- Built using **Tkinter**  
- User inputs:
  - Present price  
  - Kilometers driven  
  - Fuel type  
  - Seller type  
  - Transmission type  
  - Number of owners  
  - Car age  
- Displays predicted selling price instantly  

---

## ▶️ How to Run the Project

### Step 1: Install dependencies
```bash
pip install pandas matplotlib seaborn scikit-learn
```
###Step 2: Run model training & analysis
```bash
python car_price_pred.py

```
###Step 3: Launch GUI (runs automatically or separately)
```bash
python tinker_based.py
```

---

## 🚀 Future Improvements

- Use advanced models (Random Forest, XGBoost)
- Save trained model using pickle or joblib
- Improve GUI design and input validation
- Deploy the application as a web-based system
