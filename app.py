import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from scipy.stats import ttest_rel
import time

# Function to select features based on Random Forest feature importance
def select_features_with_random_forest(X_train_scaled, y_train, param_grid, threshold="median"):
    model = RandomForestRegressor(random_state=42)

    # Perform Grid Search to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Select features based on importance
    selector = SelectFromModel(best_model, threshold=threshold)
    selector.fit(X_train_scaled, y_train)
    selected_features = np.where(selector.get_support())[0]

    return selected_features, selector, best_model

# Function to optimize the number of components for PLS using cross-validation
def optimize_pls_components(X, y, max_components=10, n_splits=5):
    best_n_components = 1
    best_cv_score = -np.inf

    for n_components in range(1, max_components + 1):
        pls = PLSRegression(n_components=n_components)
        cv_scores = cross_val_score(pls, X, y, cv=n_splits, scoring='neg_mean_squared_error')
        mean_cv_score = np.mean(cv_scores)

        if mean_cv_score > best_cv_score:
            best_cv_score = mean_cv_score
            best_n_components = n_components

    return best_n_components

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

    # Prepare the data
    X = df.iloc[:, 1:]  # Spectral data
    y = df.iloc[:, 0:1]  # Concentrations of drugs A, B, and C
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 1. Feature Selection using Random Forest with Hyperparameter Tuning ---

    # Define the parameter grid for GridSearchCV (Random Forest)
    param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Select features using Random Forest with Grid Search
    start_time = time.time()
    best_features_rf, rf_selector, best_rf_model = select_features_with_random_forest(X_train_scaled, y_train, param_grid_rf)
    end_time = time.time()

    st.write("Selected Features (Indices) using Random Forest:", best_features_rf)
    st.write(f"Number of Selected Features: {len(best_features_rf)}")
    st.write("Best Random Forest Model:", best_rf_model)
    st.write(f"Feature selection with Random Forest took {end_time - start_time:.2f} seconds")

    # --- 2. Build and Evaluate PLS Model ---

    # Optimize n_components for PLS without feature selection
    best_n_components_without_fs = optimize_pls_components(X_train_scaled, y_train, max_components=10)
    st.write(f"\nBest number of components for PLS without feature selection: {best_n_components_without_fs}")

    # Without Feature Selection
    pls_without_fs = PLSRegression(n_components=best_n_components_without_fs)
    pls_without_fs.fit(X_train_scaled, y_train)

    # Predictions on training and testing sets
    y_train_pred_without_fs = pls_without_fs.predict(X_train_scaled)
    y_test_pred_without_fs = pls_without_fs.predict(X_test_scaled)

    # Calculate metrics for training set
    mse_train_without_fs = mean_squared_error(y_train, y_train_pred_without_fs)
    mae_train_without_fs = mean_absolute_error(y_train, y_train_pred_without_fs)
    r2_train_without_fs = r2_score(y_train, y_train_pred_without_fs)

    # Calculate metrics for testing set
    mse_test_without_fs = mean_squared_error(y_test, y_test_pred_without_fs)
    mae_test_without_fs = mean_absolute_error(y_test, y_test_pred_without_fs)
    r2_test_without_fs = r2_score(y_test, y_test_pred_without_fs)

    st.write("PLS without Feature Selection (Training Set):")
    st.write("  MSE:", mse_train_without_fs)
    st.write("  MAE:", mae_train_without_fs)
    st.write("  R-squared:", r2_train_without_fs)

    st.write("PLS without Feature Selection (Testing Set):")
    st.write("  MSE:", mse_test_without_fs)
    st.write("  MAE:", mae_test_without_fs)
    st.write("  R-squared:", r2_test_without_fs)

    # Optimize n_components for PLS with feature selection (if features were selected)
    if best_features_rf.size > 0:
        X_train_selected = X_train_scaled[:, best_features_rf]
        X_test_selected = X_test_scaled[:, best_features_rf]

        best_n_components_with_fs = optimize_pls_components(X_train_selected, y_train, max_components=4)
        st.write(f"Best number of components for PLS with feature selection: {best_n_components_with_fs}")

        # With Feature Selection
        pls_with_fs = PLSRegression(n_components=best_n_components_with_fs)
        pls_with_fs.fit(X_train_selected, y_train)

        # Predictions on training and testing sets
        y_train_pred_with_fs = pls_with_fs.predict(X_train_selected)
        y_test_pred_with_fs = pls_with_fs.predict(X_test_selected)

        # Calculate metrics for training set
        mse_train_with_fs = mean_squared_error(y_train, y_train_pred_with_fs)
        mae_train_with_fs = mean_absolute_error(y_train, y_train_pred_with_fs)
        r2_train_with_fs = r2_score(y_train, y_train_pred_with_fs)

        # Calculate metrics for testing set
        mse_test_with_fs = mean_squared_error(y_test, y_test_pred_with_fs)
        mae_test_with_fs = mean_absolute_error(y_test, y_test_pred_with_fs)
        r2_test_with_fs = r2_score(y_test, y_test_pred_with_fs)

        st.write("PLS with Feature Selection (Training Set):")
        st.write("  MSE:", mse_train_with_fs)
        st.write("  MAE:", mae_train_with_fs)
        st.write("  R-squared:", r2_train_with_fs)

        st.write("PLS with Feature Selection (Testing Set):")
        st.write("  MSE:", mse_test_with_fs)
        st.write("  MAE:", mae_test_with_fs)
        st.write("  R-squared:", r2_test_with_fs)

    else:
        st.write("Warning: No features were selected by Random Forest. Skipping PLS with feature selection.")
