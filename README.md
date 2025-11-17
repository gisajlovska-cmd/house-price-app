# 🏡 California Housing Price Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b.svg)
![Scikit-learn](https://img.shields.io/badge/ML-scikit--learn-f7931e.svg)

This project is an end-to-end machine learning application that predicts **median house prices in California** based on demographic, geographic, and housing features.  
The final model is deployed as an interactive **Streamlit web app**, where users can input custom house characteristics and receive an estimated price.

---

## 📊 Project Overview

The workflow includes:

1. **Exploratory Data Analysis (EDA)**  
   - Summary statistics, distributions, and correlation heatmaps  
   - Understanding relationships between features and `median_house_value`

2. **Data Preprocessing**  
   - Handling missing values (`total_bedrooms`)  
   - One-hot encoding of the `ocean_proximity` categorical variable  
   - Feature scaling using `StandardScaler` for all numerical features

3. **Modeling**  
   Several regression models were tested:
   - Linear Regression  
   - RANSAC Regression  
   - Ridge Regression  
   - Decision Tree Regressor  
   - Random Forest Regressor ✅ (best performer)

   The **Random Forest Regressor** achieved the best performance on the test set and was chosen as the final model.

4. **Deployment**  
   - The trained model and scaler are saved to `model.pkl` using `joblib`  
   - A Streamlit app (`app.py`) loads the model and allows interactive predictions

---

## 🧠 Features Used

The model uses the following features:

- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`
- One-hot encoded categories from `ocean_proximity`:
  - `<1H OCEAN`
  - `INLAND`
  - `ISLAND`
  - `NEAR BAY`
  - `NEAR OCEAN`

**Target variable:**
- `median_house_value`

---

## 🏗 Project Structure

```text
house-price-app/
├── app.py                  # Streamlit web app
├── model.pkl               # Trained model + scaler + feature info
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── data/
│   └── housing.csv         # California housing dataset
└── src/
    └── train_model_new.ipynb   # Jupyter notebook for training & analysis
