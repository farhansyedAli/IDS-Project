import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load the trained model
pipeline = joblib.load("developer_role_model.joblib")

# LabelEncoder mapping (adjust as per your encoding)
role_label_map = {
    0: 'Backend',
    1: 'Developer',
    2: 'Engineer',
    3: 'Front End',
    4: 'Full Stack',
    5: 'Student',
    6: 'Other'
}

st.set_page_config(page_title="Developer Role Predictor", layout="wide")
st.title("🧠 Developer Role Prediction App")

# --- Sidebar Navigation ---
section = st.sidebar.radio("Go to", ["Introduction", "EDA", "Model", "Conclusion"])

# --- Introduction ---
if section == "Introduction":
    st.header("📊 Introduction")
    st.markdown("""
    This Streamlit app predicts the **developer's role** (e.g., Full Stack, Backend, Front End, etc.) based on
    their **tech stack** and **profile**.

    **Dataset Source**: Kaggle – Developers' Tech Stack

    **Project Goal**: Build a machine learning model to classify a developer's role from their description, tags, and device info.
    """)

# --- EDA ---
elif section == "EDA":
    st.header("📈 Exploratory Data Analysis (EDA)")
    df = pd.read_csv("developers_cleaned.csv")  # Preprocessed EDA dataset

    st.subheader("🔍 Dataset Snapshot")
    st.dataframe(df.head())

    st.subheader("📦 Role Distribution")
    role_counts = df['Role'].value_counts()
    st.bar_chart(role_counts)

    st.subheader("🌐 Country Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Country', order=df['Country'].value_counts().index[:10], ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)

# --- Model ---
elif section == "Model":
    st.header("🤖 Model & Prediction")
    st.markdown("""
    We used a **Random Forest Classifier** with a pipeline that includes:
    - TF-IDF Vectorization for text (`Description`, `Tags`)
    - OneHotEncoding for categorical features (`Country`, `Computer`, `Phone`)
    - Label Encoding for target (`Role`)
    """)

    st.subheader("🔮 Make a Prediction")
    description = st.text_area("📝 Description", "Experienced developer working with Python and React.")
    tags = st.text_input("🏷️ Tags", "Python, Flask, React")
    country = st.selectbox("🌍 Country", ['USA', 'India', 'Germany', 'Brazil', 'Pakistan', 'Other'])
    computer = st.selectbox("💻 Computer", ['Windows', 'Mac', 'Linux'])
    phone = st.selectbox("📱 Phone", ['iPhone', 'Android', 'Other'])

    if st.button("Predict Developer Role"):
        input_df = pd.DataFrame([{
            'Description': description,
            'Tags': tags,
            'Country': country,
            'Computer': computer,
            'Phone': phone
        }])
        prediction = pipeline.predict(input_df)[0]
        predicted_role = role_label_map.get(prediction, "Unknown")
        st.success(f"Predicted Role: **{predicted_role}**")

    st.subheader("📉 Model Performance")
    df_perf = pd.read_csv("model_evaluation.csv")  # Pre-saved classification report metrics
    st.dataframe(df_perf)

    st.subheader("📌 Confusion Matrix")
    cm = pd.read_csv("confusion_matrix.csv", index_col=0)  # Pre-saved confusion matrix
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# --- Conclusion ---
elif section == "Conclusion":
    st.header("✅ Conclusion")
    st.markdown("""
    - Successfully built a machine learning model to classify developer roles using profile data.
    - Achieved **high accuracy (~95%)** on test data.
    - Used interactive visualizations and prediction interface.

    🚀 Future Work:
    - Improve role extraction using NLP
    - Add more user metadata (e.g., years of experience)
    - Enable bulk file upload for multi-user prediction
    """)
