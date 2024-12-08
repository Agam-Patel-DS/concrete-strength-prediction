import streamlit as st
import joblib
import numpy as np

# Title of the App
st.title("Concrete Strength Prediction")
st.subheader("PW Sklills Internship")
# Sidebar configuration
st.sidebar.header("Settings")
MODEL_PATH = "models/GradientBoosting_best_model.pkl"  # Path to the saved model

# Step 1: Load the Pre-trained Model
@st.cache_resource
def load_model():
    """Load a pre-trained model from the specified path."""
    try:
        model = joblib.load(MODEL_PATH)
        st.success(f"Model loaded successfully from {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Step 2: Concrete Strength Prediction Input via Sliders
if model is not None:
    st.header("Input Features for Prediction")

    # Concrete dataset features: 8 common features
    cement = st.slider("Cement (kg/m^3)", 0, 500, 300)  # Range based on dataset scale
    blast_furnace_slag = st.slider("Blast Furnace Slag (kg/m^3)", 0, 200, 100)
    fly_ash = st.slider("Fly Ash (kg/m^3)", 0, 200, 50)
    water = st.slider("Water (kg/m^3)", 0, 200, 150)
    superplasticizer = st.slider("Superplasticizer (kg/m^3)", 0, 50, 10)
    coarse_aggregate = st.slider("Coarse Aggregate (kg/m^3)", 500, 1500, 1000)
    fine_aggregate = st.slider("Fine Aggregate (kg/m^3)", 500, 1000, 700)
    age = st.slider("Age (days)", 1, 365, 28)

    # Create a feature array for the input data
    input_features = np.array([cement, blast_furnace_slag, fly_ash, water,
                               superplasticizer, coarse_aggregate, fine_aggregate, age]).reshape(1, -1)

    # Step 3: Prediction
    if st.button("Predict Concrete Strength"):
        prediction = model.predict(input_features)
        st.subheader("Predicted Concrete Strength")
        st.write(f"The predicted concrete strength is: **{prediction[0]:.2f} MPa**")
else:
    st.warning("Model is not loaded. Please ensure the model is in the correct folder.")
