import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    housing_df = pd.read_csv("C:/Users/jeffl/OneDrive/Desktop/googleA/script/Cleaned_Housing_data.csv")
    demo_df = pd.read_csv("C:/Users/jeffl/OneDrive/Desktop/googleA/script/synthetic_demographic_data.csv")
    return housing_df, demo_df

housing_df, demo_df = load_data()

# ---------------------------
# Preprocess Housing Data
# ---------------------------
def preprocess_housing(df):
    df = df.copy()
    for col in ['mainroad', 'guestroom', 'hotwaterheating', 'airconditioning']:
        if col in df.columns:
            df[col] = df[col].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, columns=['Location', 'furnishingstatus', 'property_type', 'Facilities'], drop_first=True)
    return df

housing_encoded = preprocess_housing(housing_df)

# ---------------------------
# Transformation
# ---------------------------
demo_df['income_level'] = np.log1p(demo_df['income_level'])
housing_df['price'] = np.log1p(housing_df['price'])
housing_df['area'] = np.log1p(housing_df['area'])

# ---------------------------
# Features for Matching
# ---------------------------
feature_cols = ['price', 'area', 'bedrooms', 'bathrooms', 'parking']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(housing_encoded[feature_cols])

# ---------------------------
# Demographic encoding
# ---------------------------
demo_features = ['income_level', 'fam_composition', 'age', 'occupation', 'location_preference']

occupation_encoder = LabelEncoder()
demo_df['occupation_encoded'] = occupation_encoder.fit_transform(demo_df['occupation'])

location_preference_encoder = LabelEncoder()
demo_df['location_preference_encoded'] = location_preference_encoder.fit_transform(demo_df['location_preference'])

# ---------------------------
# Classification
# ---------------------------
X_demo = demo_df[['income_level', 'fam_composition', 'age', 'occupation_encoded', 'location_preference_encoded']]

label_encoder = LabelEncoder()
housing_df['property_type_encoded'] = label_encoder.fit_transform(housing_df['property_type'])
y_class = housing_df['property_type_encoded']

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_demo, y_class, test_size=0.3, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_clf, y_train_clf)

# ---------------------------
# Regression
# ---------------------------
y_reg = housing_df['price']
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_demo, y_reg, test_size=0.3, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train_reg, y_train_reg)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Property Recommender", layout="centered")
st.title("🏡 AI Property Recommender")

income = st.number_input("Enter your annual income (RM)", min_value=10000, max_value=500000, value=60000)
gender = st.selectbox("Select gender", ["Male", "Female"])
occupation = st.selectbox("Select occupation", demo_df['occupation'].unique())
preferred_location = st.selectbox("Preferred location", housing_df['Location'].unique())
fam_composition = st.slider("Number of people in your family", min_value=1, max_value=6, value=3)
age = st.number_input("Enter your age", min_value=18, max_value=100, value=30)

if st.button("Recommend Properties"):
    with st.spinner("Finding properties..."):

        occ_encoded = occupation_encoder.transform([occupation])[0]
        loc_encoded = location_preference_encoder.transform([preferred_location])[0]

        user_demo = pd.DataFrame([{
            'income_level': np.log1p(income),
            'fam_composition': fam_composition,
            'age': age,
            'occupation_encoded': occ_encoded,
            'location_preference_encoded': loc_encoded
        }])

        predicted_class_encoded = clf.predict(user_demo)[0]
        predicted_property_type = label_encoder.inverse_transform([predicted_class_encoded])[0]

        predicted_price = np.expm1(regressor.predict(user_demo)[0])

        user_profile = {
            'price': np.log1p(income * 0.3),
            'area': np.log1p(fam_composition * 700),
            'bedrooms': min(5, max(1, fam_composition - 2)),
            'bathrooms': min(3, max(1, fam_composition // 3)),
            'parking': 1 if occupation not in ['Others'] else 0
        }

        user_df = pd.DataFrame([user_profile])
        user_scaled = scaler.transform(user_df)

        similarity = cosine_similarity(user_scaled, scaled_features)
        housing_df['similarity_score'] = similarity[0]

        filtered_df = housing_df[housing_df['property_type'] == predicted_property_type].copy()

        location_match = filtered_df['Location'] == preferred_location
        filtered_df.loc[location_match, 'similarity_score'] += 0.05

        recommended = filtered_df.sort_values(by='similarity_score', ascending=False).head(5)
        recommended['price'] = np.expm1(recommended['price'])
        recommended['area'] = np.expm1(recommended['area'])

        st.subheader("🏘 Top Property Recommendations")
        st.dataframe(recommended[['price', 'area', 'bedrooms', 'bathrooms', 'property_type', 'Location', 'similarity_score']])

        st.markdown(f"*Predicted suitable property type:* `{predicted_property_type}`")
        st.markdown(f"*Estimated suitable price:* `$ {predicted_price:,.2f}`")
