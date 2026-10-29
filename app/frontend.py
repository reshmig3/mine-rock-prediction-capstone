import os

import pandas as pd
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="SonarSense | Mine vs Rock", page_icon="🌊", layout="wide")

st.title("🌊 SonarSense: Mine vs Rock Classifier")
st.write(
    "Enter 60 sonar measurements (comma-separated). The model classifies the sample "
    "as a **Mine** or a **Rock** and shows its estimated probabilities."
)


def fetch_metrics():
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=10)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


tab_predict, tab_model = st.tabs(["Predict", "Model performance"])

with tab_predict:
    with st.form("prediction_form"):
        text = st.text_area(
            "Sonar measurements",
            placeholder="0.02, 0.03, 0.04, ... (exactly 60 numbers)",
            height=150,
        )
        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            values = [float(v) for v in text.replace("\n", ",").split(",") if v.strip()]
        except ValueError:
            st.error("Only numbers separated by commas are allowed.")
            values = None

        if values is not None:
            if len(values) != 60:
                st.error(f"Please enter exactly 60 measurements. You entered {len(values)}.")
            else:
                try:
                    r = requests.post(f"{API_BASE}/predict", json={"features": values}, timeout=60)
                    if r.ok:
                        result = r.json()
                        if result["prediction"] == "Mine":
                            st.error(f"Prediction: **Mine** ⚠️")
                        else:
                            st.success(f"Prediction: **Rock** 🪨")
                        st.write("Estimated probabilities (from the model, not guaranteed accuracy):")
                        st.bar_chart(pd.Series(result["probabilities"]))
                    elif r.status_code == 422:
                        st.error("The API rejected the input. Check that all 60 values are valid numbers.")
                    else:
                        st.error("The prediction service had a problem. Please try again in a moment.")
                except requests.RequestException:
                    st.error(
                        "Could not reach the prediction service. It may be waking up "
                        "(free hosting can take about a minute). Please try again."
                    )

with tab_model:
    metrics = fetch_metrics()
    if not metrics:
        st.info("Model metrics are not available right now.")
    else:
        st.subheader("Held-out test results")
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