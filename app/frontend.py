
import streamlit as st
import requests

st.set_page_config(
    page_title="Mine vs Rock Predictor",
    page_icon="🌊",
    layout="centered"
)

st.title("🌊 Mine vs Rock Prediction")
st.write(
    "Enter the 60 sonar measurements to predict "
    "whether the object is a Mine or a Rock."
)

API_URL = "http://127.0.0.1:8000/predict"

with st.form("prediction_form"):
    measurements_text = st.text_area(
        "Sonar measurements",
        placeholder="Enter 60 comma-separated numbers",
        height=150
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        values = [
            float(value.strip())
            for value in measurements_text.split(",")
            if value.strip()
        ]

        if len(values) != 60:
            st.error(
                f"Please enter exactly 60 measurements. "
                f"You entered {len(values)}."
            )
        else:
            response = requests.post(
                API_URL,
                json={"features": values},
                timeout=30
            )

            if response.ok:
                result = response.json()
                prediction = result["prediction"]

                if prediction == "Mine":
                    st.error("Prediction: Mine ⚠️")
                else:
                    st.success("Prediction: Rock 🪨")

                st.json(result)
            else:
                st.error(
                    f"API error: {response.status_code} "
                    f"- {response.text}"
                )

    except ValueError:
        st.error("Please enter only valid numbers separated by commas.")
    except requests.RequestException as error:
        st.error(
            "Could not connect to the API. "
            "Make sure your FastAPI server is running."
        )
        st.caption(str(error))
