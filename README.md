# Stock Price Prediction (NIFTY 50)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-orange)](https://www.kaggle.com/datasets/ashishjangra27/nifty-50-25-yrs-data)

## 📊 Overview

This project focuses on predicting **NIFTY 50 stock prices** using both **Machine Learning (ML)** and **Deep Learning (DL)** approaches. It is designed for educational purposes and provides a beginner-friendly introduction to stock market data analysis and time-series forecasting.

The workflow involves preparing time-series data, training multiple models, and comparing their performance to identify the best predictors for stock prices.

## 🚀 Features

- **Hybrid Modeling**: Combines classical ML models with neural networks for comprehensive analysis.
- **Time-Series Forecasting**: Uses sliding windows (30–250 days) to create supervised learning datasets.
- **Multiple Targets**: Predicts `Open`, `Close`, `High`, and `Low` prices.
- **Model Comparison**: Evaluates models using MAE and RMSE metrics.
- **Visualization**: Includes charts for performance analysis and insights.
- **Model Persistence**: Saves trained models and results for future use.

## 📈 Pipeline Overview

1. **Data Loading**
   - Load stock price data from `data.csv`.
   - Features: `Open`, `Close`, `High`, `Low`.

2. **Data Preparation**
   - Create supervised learning datasets using sliding windows (30–250 days).
   - Generate `(X, y)` pairs for each feature.

3. **Modeling**
   - **Machine Learning Models**:
     - Linear: `LinearRegression`, `Ridge`, `Lasso`
     - Tree-based: `RandomForest`, `GradientBoosting`, `XGBoost`, `LightGBM`
     - Others: `SVR`, `KNN`
   - **Deep Learning Models**:
     - RNN, LSTM, GRU, Bidirectional LSTM (using Keras Sequential API)

4. **Training**
   - Train models on rolling window datasets.
   - Evaluate using **MAE** and **RMSE**.

5. **Evaluation & Comparison**
   - Store results for all models.
   - Compare ML vs DL models for different input window sizes.

## 🏆 Key Highlights

- Hybrid pipeline combining **classical ML** and **neural networks**.
- Uses **multiple time horizons (30–250 days)** for robust prediction.
- Tracks **training and testing errors** to evaluate generalization.
- Identifies top-performing models for specific targets and time windows.

## 🛠️ Requirements

- Python 3.x
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `lightgbm`
  - `tensorflow` (for DL models)
  - `matplotlib`
  - `joblib`
  - `tqdm`



## 📖 Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Prepare Data**:
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ashishjangra27/nifty-50-25-yrs-data) and place `data.csv` in the root directory.

3. **Run the Notebook**:
   - Open `Stock_Price_Prediction_NIfty_50.ipynb` in Jupyter Notebook or JupyterLab.
   - Execute the cells to load data, train models, and evaluate results.

4. **Key Outputs**:
   - `models.csv`: CSV file with model performance metrics.
   - `trained_models.joblib`: Serialized trained models for inference.
   - Visualizations: Charts comparing model performance.

5. **Model Inference**:
   - Load saved models using `joblib.load('trained_models.joblib')`.
   - Use the model for predictions on new data.

## 📞 Contact

- **Author**: Rohan paswan
- **Email**:rohanpaswan001782@gmail.com
