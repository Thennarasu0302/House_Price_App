import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import logging
import io

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="House Price Prediction App", layout="wide")

# Title and description
st.title("House Price Prediction and Classification App")
st.markdown("""
This application predicts house prices (regression) and classifies whether prices are above or below the median (classification).
Upload a CSV file with house price data and select a model from the sidebar to view performance metrics and visualizations.
""")

# Check if dependencies are installed
try:
    import matplotlib
    import sklearn
except ImportError as e:
    logger.error(f"Dependency error: {str(e)}")
    st.error(f"Dependency error: {str(e)}. Please ensure all dependencies are installed via requirements.txt.")
    st.stop()

# File uploader with size limit
st.subheader("Upload Data")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

if uploaded_file is not None:
    if uploaded_file.size > MAX_FILE_SIZE:
        logger.error("Uploaded file exceeds size limit")
        st.error("Error: File size exceeds 100MB limit. Please upload a smaller file.")
        st.stop()

# Load and validate uploaded data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is None:
        logger.warning("No file uploaded")
        st.warning("Please upload a CSV file to proceed.")
        return None
    try:
        # Read CSV from uploaded file with encoding handling
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        if df.empty:
            logger.error("Uploaded CSV is empty")
            st.error("Error: The uploaded CSV file is empty.")
            return None
        logger.info("Data loaded successfully")
        return df
    except UnicodeDecodeError:
        logger.error("Invalid CSV encoding")
        st.error("Error: The CSV file has an unsupported encoding. Please ensure it is UTF-8 encoded.")
        return None
    except pd.errors.EmptyDataError:
        logger.error("Uploaded CSV is empty or invalid")
        st.error("Error: The uploaded CSV file is empty or invalid.")
        return None
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data only if file is uploaded
df = load_data(uploaded_file)

# Proceed only if data is loaded
if df is not None:
    # Display data preview
    st.subheader("Data Preview")
    st.write("First 5 rows of the uploaded data:")
    st.dataframe(df.head())

    # Preprocessing
    def preprocess_data(df):
        # Check for required column
        if "price" not in df.columns:
            logger.error("Price column missing from dataset")
            st.error("Error: 'price' column is missing from the dataset. Required column: 'price'.")
            st.stop()
        
        # Drop unnecessary columns
        df_cleaned = df.drop(columns=["date", "street", "city", "statezip", "country"], errors='ignore')
        
        # Check data size before dropping NA
        original_size = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        if len(df_cleaned) == 0:
            logger.error("No data remains after removing missing values")
            st.error("Error: No data remains after removing missing values.")
            st.stop()
        if len(df_cleaned) < original_size * 0.5:
            st.warning(f"Warning: {original_size - len(df_cleaned)} rows ({(original_size - len(df_cleaned)) / original_size * 100:.1f}%) were removed due to missing values.")
            logger.warning(f"Dropped {original_size - len(df_cleaned)} rows due to missing values")
        
        return df_cleaned

    df_cleaned = preprocess_data(df)

    # Create classification target (above/below median price)
    try:
        median_price = df_cleaned["price"].median()
        df_cleaned["price_class"] = (df_cleaned["price"] > median_price).astype(int)
    except Exception as e:
        logger.error(f"Error creating classification target: {str(e)}")
        st.error(f"Error creating classification target: {str(e)}")
        st.stop()

    # Feature-target split
    try:
        X = df_cleaned.drop(["price", "price_class"], axis=1)
        y_reg = df_cleaned["price"]
        y_clf = df_cleaned["price_class"]
    except Exception as e:
        logger.error(f"Error in feature-target split: {str(e)}")
        st.error(f"Error in feature-target split: {str(e)}")
        st.stop()

    # Check for non-numeric features
    non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(non_numeric_cols) > 0:
        logger.error(f"Non-numeric columns found: {', '.join(non_numeric_cols)}")
        st.error(f"Error: Non-numeric columns found: {', '.join(non_numeric_cols)}. Please encode categorical variables or ensure all features are numeric.")
        st.stop()

    # Check for sufficient data
    if len(X) < 10:
        logger.error("Insufficient data for modeling")
        st.error("Error: The dataset is too small for modeling (fewer than 10 samples).")
        st.stop()

    # Single train-test split for consistency
    try:
        X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
        _, _, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)  # Same split for classification
    except Exception as e:
        logger.error(f"Error in train-test split: {str(e)}")
        st.error(f"Error in train-test split: {str(e)}")
        st.stop()

    # Feature scaling (fit on training data only)
    scaler = StandardScaler()
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    except Exception as e:
        logger.error(f"Error in feature scaling: {str(e)}")
        st.error(f"Error in feature scaling: {str(e)}")
        st.stop()

    # Classification models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_name = st.sidebar.selectbox("Choose a model", list(models.keys()))

    # Train model (no caching due to unhashable model objects)
    def train_model(model, X_train, y_train, model_name):
        try:
            model.fit(X_train, y_train)
            logger.info(f"Model {model_name} trained successfully")
            return model
        except Exception as e:
            logger.error(f"Error training model {model_name}: {str(e)}")
            st.error(f"Error training model {model_name}: {str(e)}")
            st.stop()

    # Train and evaluate selected model
    results = {}
    selected_model = models[model_name]
    trained_model = train_model(selected_model, X_train_scaled, y_train_clf, model_name)
    try:
        preds_clf = trained_model.predict(X_test_scaled)
        probs_clf = trained_model.predict_proba(X_test_scaled)[:, 1]
    except Exception as e:
        logger.error(f"Error in model prediction: {str(e)}")
        st.error(f"Error in model prediction: {str(e)}")
        st.stop()

    # Classification metrics
    try:
        accuracy = accuracy_score(y_test_clf, preds_clf)
        f1 = f1_score(y_test_clf, preds_clf)
        roc_auc = roc_auc_score(y_test_clf, probs_clf)
        cm = confusion_matrix(y_test_clf, preds_clf)
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        st.error(f"Error calculating metrics: {str(e)}")
        st.stop()

    results[model_name] = {
        "model": trained_model,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm,
        "probs": probs_clf,
        "preds_clf": preds_clf
    }

    # Display results
    st.header(f"Results for {model_name}")

    # Classification metrics
    st.subheader("Classification Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{results[model_name]['accuracy']:.4f}")
    col2.metric("F1 Score", f"{results[model_name]['f1']:.4f}")
    col3.metric("ROC AUC", f"{results[model_name]['roc_auc']:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(results[model_name]['cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"Confusion Matrix ({model_name})")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

    # ROC Curve
    st.subheader("ROC Curve")
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    fpr, tpr, _ = roc_curve(y_test_clf, results[model_name]['probs'])
    ax_roc.plot(fpr, tpr, label=f"ROC Curve (AUC = {results[model_name]['roc_auc']:.4f})")
    ax_roc.plot([0, 1], [0, 1], 'r--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve ({model_name})")
    ax_roc.legend()
    ax_roc.grid(True)
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    # Regression model (using RandomForestRegressor for regression)
    st.subheader("Regression Metrics (Random Forest)")
    @st.cache_resource
    def train_regressor(X_train, y_train):
        try:
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X_train, y_train)
            logger.info("Regression model trained successfully")
            return rf_reg
        except Exception as e:
            logger.error(f"Error training regression model: {str(e)}")
            st.error(f"Error training regression model: {str(e)}")
            st.stop()

    rf_reg = train_regressor(X_train_scaled, y_train_reg)
    try:
        preds_reg = rf_reg.predict(X_test_scaled)
        mse = mean_squared_error(y_test_reg, preds_reg)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds_reg)
    except Exception as e:
        logger.error(f"Error in regression prediction or metrics: {str(e)}")
        st.error(f"Error in regression prediction or metrics: {str(e)}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R^2", f"{r2:.4f}")

    # Actual vs Predicted Prices
    st.subheader("Actual vs Predicted Prices (Regression)")
    fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test_reg, y=preds_reg, alpha=0.5, ax=ax_reg)
    ax_reg.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
    ax_reg.set_xlabel("Actual Prices")
    ax_reg.set_ylabel("Predicted Prices")
    ax_reg.set_title("Actual vs Predicted House Prices (Random Forest)")
    ax_reg.grid(True)
    st.pyplot(fig_reg)
    plt.close(fig_reg)

    # Feature Importance (for Random Forest Classifier)
    if model_name == "Random Forest":
        st.subheader("Feature Importance")
        try:
            importances = results[model_name]['model'].feature_importances_
            feature_names = X.columns
            feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
            feature_importance_df = feature_importance_df.sort_values("Importance", ascending=False)
            
            fig_fi, ax_fi = plt.subplots(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=feature_importance_df, ax=ax_fi)
            ax_fi.set_title("Feature Importance (Random Forest)")
            st.pyplot(fig_fi)
            plt.close(fig_fi)
        except Exception as e:
            logger.error(f"Error in feature importance plot: {str(e)}")
            st.error(f"Error in feature importance plot: {str(e)}")
            st.stop()
