import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import ttest_rel
import time
import streamlit as st

# --- Feature Selection using Elastic Net with Hyperparameter Tuning ---

# Function to select features based on the best Elastic Net model
def select_features_with_elastic_net(X_train_scaled, y_train, param_grid):
    # Create and fit the Elastic Net model
    model = ElasticNet(max_iter=10000, tol=1e-3)

    # Perform Grid Search to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Select features based on non-zero coefficients
    best_features = np.where(best_model.coef_ != 0)[0]

    # Get the best hyperparameters
    best_alpha = best_model.alpha
    best_l1_ratio = best_model.l1_ratio

    return best_features, best_model, best_alpha, best_l1_ratio

# Define the parameter grid for GridSearchCV (focus on Elastic Net)
param_grid_elastic_net = {
    'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # Range of l1_ratio values
}

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

    # Select features using Elastic Net with Grid Search
    start_time = time.time()
    best_features_en, best_en_model, best_alpha, best_l1_ratio = select_features_with_elastic_net(X_train_scaled, y_train, param_grid_elastic_net)
    end_time = time.time()

    st.write("Selected Features (Indices) using Elastic Net:", best_features_en)
    st.write(f"Number of Selected Features: {len(best_features_en)}")
    st.write("Best Elastic Net Model:", best_en_model)
    st.write(f"Feature selection with Elastic Net took {end_time - start_time:.2f} seconds")

    # --- 2. Build and Evaluate PLS Model ---

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

    # Optimize n_components for PLS without feature selection
    best_n_components_without_fs = optimize_pls_components(X_train_scaled, y_train)

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

    # Optimize n_components for PLS with feature selection (if features were selected)
    if best_features_en.size > 0:
        X_train_selected = X_train_scaled[:, best_features_en]
        X_test_selected = X_test_scaled[:, best_features_en]

        best_n_components_with_fs = optimize_pls_components(X_train_selected, y_train, max_components=4)

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

    # --- 3. Display Results in Two Columns ---

    col1, col2 = st.columns(2)  # Create two columns

    with col1:
        st.write("### PLS without Feature Selection")
        st.write(f"Best number of components: {best_n_components_without_fs}")
        st.write("**Training Set:**")
        st.write(f"  MSE: {mse_train_without_fs:.4f}")
        st.write(f"  MAE: {mae_train_without_fs:.4f}")
        st.write(f"  R-squared: {r2_train_without_fs:.4f}")
        st.write("**Testing Set:**")
        st.write(f"  MSE: {mse_test_without_fs:.4f}")
        st.write(f"  MAE: {mae_test_without_fs:.4f}")
        st.write(f"  R-squared: {r2_test_without_fs:.4f}")

    with col2:
        st.write("### PLS with Feature Selection (Elastic Net)")
        if best_features_en.size > 0:
            st.write(f"Best number of components: {best_n_components_with_fs}")
            st.write("**Training Set:**")
            st.write(f"  MSE: {mse_train_with_fs:.4f}")
            st.write(f"  MAE: {mae_train_with_fs:.4f}")
            st.write(f"  R-squared: {r2_train_with_fs:.4f}")
            st.write("**Testing Set:**")
            st.write(f"  MSE: {mse_test_with_fs:.4f}")
            st.write(f"  MAE: {mae_test_with_fs:.4f}")
            st.write(f"  R-squared: {r2_test_with_fs:.4f}")
        else:
            st.write("Warning: No features were selected by Elastic Net.")

else:
    st.write("Please upload a CSV file to begin analysis.")
