
import os
import streamlit as st
import requests
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="SonarSense | Mine vs Rock",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = os.getenv(
    "API_URL",
    "https://mine-rock-prediction-capstone.onrender.com/predict",
)

# ---------------- DESIGN ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #F7FBFE;
    color: #183D5B;
}
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.hero {
    background: linear-gradient(120deg, #E5F4FD 0%, #F4FAFE 65%, #D9EDFA 100%);
    border: 1px solid #D5E9F7;
    border-radius: 24px;
    padding: 36px 38px;
    margin-bottom: 24px;
}
.eyebrow {
    color: #397BA8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: clamp(30px, 4vw, 45px);
    line-height: 1.15;
    color: #183D5B;
    margin: 12px 0;
    font-weight: 800;
}
.hero p {
    color: #52718A;
    font-size: 16px;
    max-width: 680px;
    line-height: 1.7;
}
.section-title {
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
    font-size: 22px;
    color: #183D5B;
    margin: 8px 0 4px 0;
}
.section-subtitle {
    color: #6B879C;
    font-size: 14px;
    margin-bottom: 18px;
}
.info-card {
    background: white;
    border: 1px solid #DCEAF4;
    border-radius: 16px;
    padding: 20px;
    min-height: 125px;
}
.info-label {
    color: #64839A;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.info-value {
    color: #183D5B;
    font-family: 'Manrope', sans-serif;
    font-size: 21px;
    font-weight: 800;
    margin-top: 10px;
}
div[data-testid="stForm"] {
    background: #FFFFFF;
    border: 1px solid #DCEAF4;
    border-radius: 18px;
    padding: 24px;
}
.stTextArea textarea {
    background: #F8FCFF;
    border: 1px solid #CFE3F1;
    border-radius: 12px;
    color: #183D5B;
    font-size: 14px;
}
.stButton button, .stFormSubmitButton button {
    background: #397BA8;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
    transition: 0.2s ease;
}
.stButton button:hover, .stFormSubmitButton button:hover {
    background: #285F85;
    color: white;
    border: none;
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}
.footer {
    border-top: 1px solid #DCEAF4;
    margin-top: 36px;
    padding-top: 18px;
    color: #7891A4;
    font-size: 12px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">SONARSENSE · MACHINE LEARNING</div>
    <h1>Understand what lies<br>beneath the surface.</h1>
    <p>
        An interactive sonar classification application that uses a trained
        machine learning model to classify a 60-measurement sonar sample
        as either a Mine or a Rock.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- OVERVIEW ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Input format</div>
        <div class="info-value">60 measurements</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Prediction engine</div>
        <div class="info-value">ML classification</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">Inference service</div>
        <div class="info-value">Live API</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.markdown('<div class="section-title">Prediction workspace</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Enter a real sonar sample containing exactly 60 numerical measurements.</div>',
    unsafe_allow_html=True,
)

# ---------------- INPUT + PREDICTION ----------------
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    with st.form("prediction_form"):
        measurements_text = st.text_area(
            "Sonar measurements",
            placeholder="Paste 60 comma-separated numbers here...",
            height=210,
            help="Enter exactly 60 numeric values separated by commas.",
        )
        submitted = st.form_submit_button(
            "Run prediction  →",
            use_container_width=True,
        )

    st.caption(
        "Please use measurements from a valid sonar sample. "
        "Arbitrary values may produce meaningless predictions."
    )

with right:
    st.markdown("""
    <div class="info-card">
        <div class="info-label">How it works</div>
        <div class="info-value">Input → Model → Result</div>
        <p style="color:#6B879C;line-height:1.7;font-size:14px;">
            Your measurements are sent to the deployed prediction API.
            The trained model processes the input and returns its predicted
            class. The application does not invent a confidence score.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Keep a history for this browser session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if submitted:
    try:
        values = [
            float(item.strip())
            for item in measurements_text.split(",")
            if item.strip()
        ]

        if len(values) != 60:
            st.error(
                f"You entered {len(values)} measurements. "
                "Exactly 60 numerical measurements are required."
            )
        elif not all(pd.notna(value) for value in values):
            st.error("Please enter valid finite numerical values.")
        elif not all(abs(value) != float("inf") for value in values):
            st.error("Infinite values are not allowed.")
        else:
            with st.spinner("Sending measurements to the prediction model..."):
                response = requests.post(
                    API_URL,
                    json={"features": values},
                    timeout=60,
                )

            if response.ok:
                result = response.json()
                prediction = result.get("prediction", "Unknown")
                label = result.get("label", "")

                st.session_state.prediction_history.insert(
                    0,
                    {
                        "Prediction": prediction,
                        "Model label": label,
                    },
                )

                st.write("")
                st.markdown(
                    '<div class="section-title">Prediction result</div>',
                    unsafe_allow_html=True,
                )

                if prediction == "Mine":
                    st.error(
                        "Predicted class: MINE. "
                        "This is a model output, not a verified real-world finding."
                    )
                elif prediction == "Rock":
                    st.success("Predicted class: ROCK")
                else:
                    st.info(f"Predicted class: {prediction}")

                st.caption(
                    f"Model label returned by the API: {label or 'Not provided'}"
                )

                st.markdown("**Measurement profile**")
                chart_data = pd.DataFrame({
                    "Measurement index": range(1, 61),
                    "Value": values,
                }).set_index("Measurement index")
                st.line_chart(chart_data, height=240)

                with st.expander("View submitted measurements"):
                    st.dataframe(
                        pd.DataFrame({
                            "Measurement": range(1, 61),
                            "Value": values,
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
            else:
                st.error(
                    f"The prediction API returned HTTP {response.status_code}. "
                    "Please try again later."
                )
                st.caption(response.text[:500])

    except ValueError:
        st.error(
            "Invalid input. Enter numbers separated by commas, "
            "without words or other characters."
        )
    except requests.RequestException:
        st.error(
            "Could not connect to the prediction API. "
            "The service may be starting up. Wait a little and retry."
        )

# ---------------- HISTORY ----------------
if st.session_state.prediction_history:
    st.write("")
    st.markdown('<div class="section-title">Recent predictions</div>',
                unsafe_allow_html=True)
    st.caption("Session-only history; it clears when the session resets.")

    st.dataframe(
        pd.DataFrame(st.session_state.prediction_history),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Clear prediction history"):
        st.session_state.prediction_history = []
        st.rerun()

# ---------------- METHODOLOGY ----------------
st.write("")
with st.expander("About the model and its limitations"):
    st.markdown("""
    **Objective:** Classify sonar measurement samples into Mine or Rock.

    **Workflow:** A user submits 60 numeric measurements. The frontend sends
    them to a FastAPI service, which loads the serialized machine learning
    model and returns a predicted class.

    **Important limitations:**
    - A prediction is not proof that an object is a mine or a rock.
    - Model accuracy and confidence are not displayed because they have not
      been verified for this interface.
    - Predictions depend on the input data and the quality of the trained model.
    - This educational project must not be used for operational mine detection
      or safety-critical decisions.
    """)

st.markdown("""
<div class="footer">
    SONARSENSE · End-to-end machine learning deployment project<br>
    Built with Streamlit, FastAPI and scikit-learn
</div>
""", unsafe_allow_html=True)
