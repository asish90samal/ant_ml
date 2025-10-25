import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------------
# Load trained model
# --------------------------------------------------------
model = joblib.load("models/best_model.pkl")

# --------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------
st.title("📡 Fractal Antenna Selector")
st.write("This app predicts the **best antenna type** based on environment and signal conditions.")

# --------------------------------------------------------
# User Inputs
# --------------------------------------------------------
env = st.selectbox("Environment Type", ["urban", "rural", "indoor"])

distance = st.number_input(
    "Distance (m)",
    min_value=1.0,
    max_value=1000.0,
    value=100.0,
    step=1.0
)
if distance > 500:
    st.warning("⚠️ Distance beyond 500 m — prediction may be less accurate.")

walls = st.number_input(
    "Number of Walls",
    min_value=0,
    max_value=40,
    value=0,
    step=1
)
if walls > 10:
    st.warning("⚠️ Number of walls is outside the training range — prediction may be less accurate.")

metal = st.selectbox("Has Metal Obstructions?", ["No", "Yes"])
metal = 1 if metal == "Yes" else 0

freq = st.number_input(
    "Frequency (MHz)",
    min_value=0.0,
    max_value=6000.0,
    value=2400.0,
    step=10.0
)
if freq > 2400:
    st.warning("⚠️ Frequency beyond 2400 MHz — prediction may be less accurate.")

# --------------------------------------------------------
# Prepare input for model
# --------------------------------------------------------
df_input = pd.DataFrame([{
    "env_type": env,
    "distance_m": distance,
    "num_walls": walls,
    "has_metal": metal,
    "frequency_mhz": freq
}])

# --------------------------------------------------------
# Prediction Logic
# --------------------------------------------------------
if st.button("🔍 Predict Best Antenna"):
    pred = model.predict(df_input)[0]

    antenna_map = {
        0: ("Sierpinski Fractal Antenna", "designs/sierpinski.png"),
        1: ("Koch Fractal Antenna", "designs/koch.png"),
        2: ("Monopole Antenna", "designs/monopole.png")
    }

    antenna_name, antenna_image = antenna_map[pred]

    st.success(f"✅ **Best Antenna:** {antenna_name}")

    # Display antenna image if it exists
    if os.path.exists(antenna_image):
        st.image(antenna_image, caption=antenna_name, use_container_width=True)
    else:
        st.warning(f"⚠️ Image not found: {antenna_image}")

    st.write("### Input Summary:")
    st.dataframe(df_input)

# --------------------------------------------------------
# Footer
# --------------------------------------------------------
st.markdown("---")
st.caption("Developed as part of the *AI-based Fractal Antenna Selection* project.")
