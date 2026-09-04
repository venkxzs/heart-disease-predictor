import streamlit as st
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("❤️ Heart Disease Prediction System")

st.write(
    "Enter the patient-related values below to generate "
    "a machine-learning prediction."
)

st.divider()


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = load_breast_cancer()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = data.target

    return X, y


X, y = load_data()


# =========================================================
# TRAIN MODEL
# =========================================================

@st.cache_resource
def train_model(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    accuracy = model.score(
        X_test,
        y_test
    )

    return model, accuracy


model, accuracy = train_model(X, y)


# =========================================================
# MODEL INFORMATION
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Dataset Records",
        len(X)
    )

with col2:
    st.metric(
        "Model",
        "Random Forest"
    )

with col3:
    st.metric(
        "Test Accuracy",
        f"{accuracy:.2%}"
    )


st.divider()


# =========================================================
# USER INPUT
# =========================================================

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)


with col1:

    mean_radius = st.number_input(
        "Mean Radius",
        value=14.0
    )

    mean_texture = st.number_input(
        "Mean Texture",
        value=19.0
    )

    mean_perimeter = st.number_input(
        "Mean Perimeter",
        value=90.0
    )

    mean_area = st.number_input(
        "Mean Area",
        value=650.0
    )


with col2:

    mean_smoothness = st.number_input(
        "Mean Smoothness",
        value=0.09,
        format="%.5f"
    )

    mean_compactness = st.number_input(
        "Mean Compactness",
        value=0.10,
        format="%.5f"
    )

    mean_concavity = st.number_input(
        "Mean Concavity",
        value=0.08,
        format="%.5f"
    )

    mean_concave_points = st.number_input(
        "Mean Concave Points",
        value=0.05,
        format="%.5f"
    )



# =========================================================
# PREDICTION
# =========================================================

st.divider()


if st.button(
    "❤️ Predict",
    type="primary",
    use_container_width=True
):

    selected_columns = [
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
        "mean compactness",
        "mean concavity",
        "mean concave points"
    ]


    input_values = [
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        mean_compactness,
        mean_concavity,
        mean_concave_points
    ]


    # Create full input dataframe
    full_input = pd.DataFrame(
        np.zeros((1, len(X.columns))),
        columns=X.columns
    )


    # Add entered values
    for column, value in zip(
        selected_columns,
        input_values
    ):

        full_input[column] = value


    # Prediction
    prediction = model.predict(
        full_input
    )[0]


    probability = model.predict_proba(
        full_input
    )[0]


    confidence = float(
        np.max(probability)
    )


    # =====================================================
    # RESULT
    # =====================================================

    st.subheader("📊 Prediction Result")

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        if prediction == 1:

            st.success(
                "✅ Prediction: Low Risk / Negative Prediction"
            )

        else:

            st.error(
                "⚠️ Prediction: High Risk / Positive Prediction"
            )


    with result_col2:

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2%}"
        )


    st.info(
        f"Model Test Accuracy: {accuracy:.2%}"
    )


# =========================================================
# DATASET
# =========================================================

with st.expander("📊 View Dataset"):

    st.dataframe(
        X,
        use_container_width=True
    )
    