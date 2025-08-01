# app_complete.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Streamlit config
st.set_page_config(page_title="🏘 AI Housing App - Regression & Classification", layout="wide")
st.title("🏘 AI Housing App - Regression & Classification")

# Load and preprocess data
@st.cache_data
def load_data():
    demo = pd.read_csv("synthetic_demographic_data.csv")
    house = pd.read_csv("Cleaned_Housing_data.csv")
    df = pd.concat([demo, house], axis=1)

    # Fill missing numerical with mean
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # Fill categorical with mode
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df

df = load_data()

# Sidebar configs
st.sidebar.header("Task Setup")
task = st.sidebar.radio("Select Task", ["Regression", "Classification"])
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2, step=0.05)
random_state = st.sidebar.number_input("Random State", 0, 200, 42)

# Feature & target setup
cat_features = ['mainroad', 'guestroom', 'hotwaterheating', 'airconditioning',
                'Location', 'furnishingstatus', 'Facilities', 'marital_status', 'location_preference']
num_features = ['income_level', 'fam_composition', 'age', 'area', 'bedrooms', 'bathrooms', 'parking']

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)

if task == "Regression":
    target = 'price'
    # Select only numeric columns and drop target
    X = df_encoded.select_dtypes(include=['number']).drop(columns=[target, 'property_type'], errors='ignore')
    y = df_encoded[target]

    st.sidebar.subheader("Regression Models")
    model_name = st.sidebar.selectbox("Choose Regressor", [
        "Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting",
        "Gaussian Process", "Neural Network"])

    # Split & scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(random_state=random_state)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=random_state)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=random_state)
    elif model_name == "Gaussian Process":
        model = GaussianProcessRegressor()
    elif model_name == "Neural Network":
        model = MLPRegressor(random_state=random_state, max_iter=1000)

    model.fit(X_train_scaled, y_train)
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    st.subheader(f"📊 {model_name} - Regression Performance")
    st.write(f"**Train R²**: {r2_score(y_train, y_pred_train):.4f}")
    st.write(f"**Test  R²**: {r2_score(y_test, y_pred_test):.4f}")
    st.write(f"**Train RMSE**: {np.sqrt(mean_squared_error(y_train, y_pred_train)):,.2f}")
    st.write(f"**Test  RMSE**: {np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")

    st.subheader("Actual vs Predicted (Test Set)")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred_test, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid()
    st.pyplot(fig)

else:  # Classification
    target = 'property_type'

    # Encode target
    le = LabelEncoder()
    df_encoded[target] = le.fit_transform(df[target])

    # Select only numeric columns and drop target
    X = df_encoded.select_dtypes(include=['number']).drop(columns=[target, 'price'], errors='ignore')
    y = df_encoded[target]

    st.sidebar.subheader("Classification Models")
    model_name = st.sidebar.selectbox("Choose Classifier", [
        "Decision Tree", "Random Forest", "Neural Network"])

    # Split & scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=random_state)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=random_state)
    elif model_name == "Neural Network":
        model = MLPClassifier(random_state=random_state, max_iter=1000)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    st.subheader(f"📊 {model_name} - Classification Performance")
    st.write(f"**Accuracy**: {accuracy_score(y_test, y_pred):.4f}")
    st.text("\nClassification Report:\n" + classification_report(y_test, y_pred, target_names=le.classes_))

# Option to show raw data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)
