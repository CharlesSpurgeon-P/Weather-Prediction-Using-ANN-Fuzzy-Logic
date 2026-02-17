# Weather Prediction using ANN and Fuzzy Logic

## Overview
This project implements a **hybrid soft computing model** for **next-day weather prediction**
by combining **Artificial Neural Networks (ANN)** with **Fuzzy Logic**.

The model learns from historical weather data and predicts multiple weather
parameters simultaneously. Fuzzy logic is used to handle uncertainty in
temperature ranges before feeding inputs to the neural network.

---

## Objectives
- Apply fuzzy logic to represent uncertainty in temperature values
- Build an ANN-based multi-output regression model
- Perform next-day weather forecasting using real-world data
- Visualize training convergence and prediction trends

---

## Features
- Hybrid ANN + Fuzzy Logic approach
- Multi-output weather prediction
- Next-day forecasting
- Real-world European weather dataset
- Training and validation loss visualization
- Actual vs predicted temperature plot

---

## Technologies Used
- Python
- TensorFlow / Keras
- scikit-learn
- scikit-fuzzy
- NumPy
- Pandas
- Matplotlib

---

## Dataset
- Historical European weather data
- Selected features:
  - Mean temperature (Basel)
  - Humidity (Basel)
  - Wind speed (De Bilt)
  - Atmospheric pressure (Basel)
- Missing values handled by removing incomplete rows

**Note:**  
Humidity and pressure values in the dataset are already provided in
normalized or fractional form. The model predicts values in the same scale
as the dataset.

---

## Model Architecture
- **Input Layer**
  - Fuzzy temperature membership values (Low, Medium, High)
  - Normalized meteorological features
- **Hidden Layers**
  - Dense layer with 12 neurons (ReLU)
  - Dense layer with 8 neurons (ReLU)
- **Output Layer**
  - Dense layer with 4 neurons (multi-output regression)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

---

## Training Details
- Train-test split: 80% / 20%
- Validation split: 20% of training data
- Epochs: 100
- Evaluation metrics:
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)

---

## Results
- Stable convergence observed during training
- Low test error indicating effective learning
- Predicted temperature trends closely follow actual values
- Model successfully captures short-term weather patterns

---

## How to Run

### 1. Install dependencies
```bash
pip install tensorflow scikit-fuzzy numpy pandas matplotlib scikit-learn
 2.Run
python weather_prediction_ann_fuzzy.py
