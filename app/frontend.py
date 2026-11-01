import math
import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="SonarSense | Mine vs Rock",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
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


def fetch_metrics():
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=15)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


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

# ---------------- OVERVIEW CARDS ----------------
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
        <div class="info-value">SVM pipeline</div>
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

tab_predict, tab_perf, tab_about = st.tabs(
    ["Predict", "Model performance", "About the model"]
)

# ---------------- PREDICT TAB ----------------
with tab_predict:
    st.markdown('<div class="section-title">Prediction workspace</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter a real sonar sample containing exactly 60 numerical measurements, separated by commas.</div>',
        unsafe_allow_html=True,
    )

    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []

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
                "Run prediction →", use_container_width=True
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
                The trained model returns its predicted class and the
                probabilities it estimated for each class. These are model
                estimates, not guaranteed accuracy.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if submitted:
        try:
            values = [
                float(item.strip())
                for item in measurements_text.split(",")
                if item.strip()
            ]
        except ValueError:
            values = None
            st.error("Invalid input. Enter numbers separated by commas, without words or other characters.")

        if values is not None:
            if len(values) != 60:
                st.error(
                    f"You entered {len(values)} measurements. "
                    "Exactly 60 numerical measurements are required."
                )
            elif not all(math.isfinite(v) for v in values):
                st.error("Please enter finite numerical values only (no infinity or NaN).")
            else:
                try:
                    with st.spinner("Sending measurements to the prediction model..."):
                        response = requests.post(
                            f"{API_BASE}/predict",
                            json={"features": values},
                            timeout=60,
                        )

                    if response.ok:
                        result = response.json()
                        prediction = result.get("prediction", "Unknown")
                        label = result.get("label", "")
                        probabilities = result.get("probabilities", {})

                        st.session_state.prediction_history.insert(
                            0, {"Prediction": prediction, "Model label": label}
                        )

                        st.write("")
                        st.markdown('<div class="section-title">Prediction result</div>',
                                    unsafe_allow_html=True)

                        if prediction == "Mine":
                            st.error("Predicted class: MINE. This is a model output, not a verified real-world finding.")
                        elif prediction == "Rock":
                            st.success("Predicted class: ROCK")
                        else:
                            st.info(f"Predicted class: {prediction}")

                        if probabilities:
                            st.markdown("**Estimated probabilities** (from the model, not guaranteed accuracy)")
                            st.bar_chart(pd.Series(probabilities))

                        st.markdown("**Measurement profile**")
                        profile = pd.DataFrame({"Value": values}, index=range(1, 61))
                        profile.index.name = "Measurement index"
                        st.line_chart(profile, height=240)

                        with st.expander("View submitted measurements"):
                            st.dataframe(
                                pd.DataFrame({"Measurement": range(1, 61), "Value": values}),
                                use_container_width=True,
                                hide_index=True,
                            )
                    elif response.status_code == 422:
                        st.error("The API rejected the input. Check that all 60 values are valid numbers.")
                    else:
                        st.error(
                            f"The prediction service returned HTTP {response.status_code}. "
                            "Please try again in a moment."
                        )
                except requests.RequestException as e:
                    st.error(f"Could not reach the prediction API: {e}")

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

# ---------------- MODEL PERFORMANCE TAB ----------------
with tab_perf:
    st.markdown('<div class="section-title">Model performance</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Verified results from the training script (training/train.py).</div>',
        unsafe_allow_html=True,
    )

    metrics = fetch_metrics()
    if not metrics:
        st.info("Model metrics are not available right now. Please try again shortly.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{metrics['test_accuracy']:.1%}")
        c2.metric("Mine precision", f"{metrics['precision_mine']:.1%}")
        c3.metric("Mine recall", f"{metrics['recall_mine']:.1%}")
        c4.metric("Mine F1", f"{metrics['f1_mine']:.1%}")

        st.subheader("Cross-validation (training split, 5 folds)")
        cv = pd.DataFrame(metrics["cross_validation"]).T
        cv["mean"] = cv["mean"].map(lambda x: f"{x:.1%}")
        cv["std"] = cv["std"].map(lambda x: f"± {x:.1%}")
        st.dataframe(cv, use_container_width=True)

        st.subheader("Confusion matrix (test set)")
        cm = pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual Mine", "Actual Rock"],
            columns=["Predicted Mine", "Predicted Rock"],
        )
        st.table(cm)

        st.caption(
            f"Selected model: {metrics['model']} · "
            f"Trained on {metrics['train_rows']} rows, tested on {metrics['test_rows']}. "
            "Small sample sizes mean these numbers carry real uncertainty."
        )

# ---------------- ABOUT TAB ----------------
with tab_about:
    st.markdown('<div class="section-title">About the model and its limitations</div>',
                unsafe_allow_html=True)
    st.markdown("""
    **Objective:** Classify sonar measurement samples into Mine or Rock.

    **Workflow:** A user submits 60 numeric measurements. The frontend sends them
    to a FastAPI service, which loads a saved scikit-learn pipeline (feature scaling
    plus an SVM classifier) and returns a predicted class with estimated probabilities.

    **Data:** UCI Sonar dataset (208 samples, 60 measurements each), from `data/sonar.csv`.

    **Important limitations:**
    - A prediction is not proof that an object is a mine or a rock.
    - The test set has only 42 samples, so the accuracy figures carry real uncertainty.
    - Probabilities are the model's estimates, not guaranteed accuracy.
    - This educational project must not be used for operational mine detection
      or safety-critical decisions.
    """)

st.markdown("""
<div class="footer">
    SONARSENSE · End-to-end machine learning deployment project<br>
    Built with Streamlit, FastAPI and scikit-learn
</div>
""", unsafe_allow_html=True)