import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
import joblib

# Load trained models (assuming they are in the same directory or a 'models' subdirectory)
try:
    lasso_model = joblib.load('lasso_model.joblib')
    pls_model_with_fs = joblib.load('pls_with_fs_model.joblib')
    pls_model_without_fs = joblib.load('pls_without_fs_model.joblib')
except FileNotFoundError:
    st.error("Model files not found. Please make sure the models are trained and saved correctly.")
    lasso_model = None
    pls_model_with_fs = None
    pls_model_without_fs = None

# Function to preprocess data
def preprocess_data(df):
    X = df.iloc[:, 1:]  # Assuming spectral data starts from column 3
    y = df.iloc[:, 0:1]  # Assuming concentrations are in the first 3 columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

# Function to perform feature selection using the loaded Lasso model
def select_features_with_lasso(X_scaled, lasso_model):
    selected_features = np.where(lasso_model.coef_ != 0)[0]
    return X_scaled[:, selected_features]

# Streamlit app
st.title("Spectral Data Analysis with PLS")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the DataFrame
    st.write("### First few rows of the uploaded data:")
    st.write(df.head())

    # Preprocess the data
    X_scaled, y = preprocess_data(df)

    # Feature selection (if Lasso model is loaded)
    if lasso_model is not None:
        X_selected = select_features_with_lasso(X_scaled, lasso_model)
        st.write(f"Number of features selected by Lasso: {X_selected.shape[1]}")
    else:
        X_selected = X_scaled  # Use all features if no model is loaded
        st.write("Lasso model not loaded. Using all features.")

    # PLS without feature selection
    if pls_model_without_fs is not None:
        y_pred_without_fs = pls_model_without_fs.predict(X_scaled)
        mse_without_fs = mean_squared_error(y, y_pred_without_fs)
        mae_without_fs = mean_absolute_error(y, y_pred_without_fs)
        r2_without_fs = r2_score(y, y_pred_without_fs)

        st.write("### PLS without Feature Selection:")
        st.write("  MSE:", mse_without_fs)
        st.write("  MAE:", mae_without_fs)
        st.write("  R-squared:", r2_without_fs)

    # PLS with feature selection
    if pls_model_with_fs is not None and X_selected.shape[1] > 0:
        y_pred_with_fs = pls_model_with_fs.predict(X_selected)
        mse_with_fs = mean_squared_error(y, y_pred_with_fs)
        mae_with_fs = mean_absolute_error(y, y_pred_with_fs)
        r2_with_fs = r2_score(y, y_pred_with_fs)

        st.write("### PLS with Feature Selection:")
        st.write("  MSE:", mse_with_fs)
        st.write("  MAE:", mae_with_fs)
        st.write("  R-squared:", r2_with_fs)
    else:
        st.write("PLS with feature selection not available (no features selected or model not loaded).")
