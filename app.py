import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv("Classification.csv")

sex_encoder = LabelEncoder()
bp_encoder = LabelEncoder()
cholesterol_encoder = LabelEncoder()
drug_encoder = LabelEncoder()

data['Sex'] = sex_encoder.fit_transform(data['Sex'])
data['BP'] = bp_encoder.fit_transform(data['BP'])
data['Cholesterol'] = cholesterol_encoder.fit_transform(data['Cholesterol'])
data['Drug'] = drug_encoder.fit_transform(data['Drug'])

# Menormaliasasikan numerik features
scaler = MinMaxScaler()
data[['Age', 'Na_to_K']] = scaler.fit_transform(data[['Age', 'Na_to_K']])

X = data.drop('Drug', axis=1)
y = data['Drug']

#training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

st.set_page_config(page_title="Aplikasi Obat Pintar", layout="wide", page_icon="🏨")

st.title("Aplikasi Obat Pintar")
st.subheader("Rekomendasi obat berdasarkan data kesehatan Anda")

# Input Form
st.sidebar.header("Masukkan Data Anda")

age = st.sidebar.slider("Usia", 0, 100, 25)
sex = st.sidebar.selectbox("Jenis Kelamin", ["Female", "Male"])
bp = st.sidebar.selectbox("Tekanan Darah", ["LOW", "NORMAL", "HIGH"])
cholesterol = st.sidebar.selectbox("Kolesterol", ["NORMAL", "HIGH"])
na_to_k = st.sidebar.slider("Rasio Natrium (Na) ke Kalium (K)", 0.0, 50.0, 15.0)

# Normalize input for Sex
if sex.lower() in ['male', 'm']:
    sex = 'M'
elif sex.lower() in ['female', 'f']:
    sex = 'F'
else:
    st.error("Input untuk Jenis Kelamin tidak valid. Gunakan 'Male', 'Female', 'M', atau 'F'.")
    st.stop()

# Preprocess Input Data
try:
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex_encoder.transform([sex])[0]],
        "BP": [bp_encoder.transform([bp])[0]],
        "Cholesterol": [cholesterol_encoder.transform([cholesterol])[0]],
        "Na_to_K": [na_to_k]
    })

    # Normalize input data
    input_data[['Age', 'Na_to_K']] = scaler.transform(input_data[['Age', 'Na_to_K']])
except ValueError as e:
    st.error(f"Error dalam memproses input: {e}")
    st.stop()

# Prediction
prediction_knn = knn.predict(input_data)[0]
prediction_gnb = gnb.predict(input_data)[0]

st.markdown("### 🩺 **Hasil Prediksi Obat**")
st.markdown(
    f"""
    <div style="background-color: #f9f9f9; padding: 15px; border-radius: 10px; border: 1px solid #ddd;">
        <h3 style="color: #4CAF50;">KNN Prediksi Obat: <strong>{drug_encoder.inverse_transform([prediction_knn])[0]}</strong></h3>
        <h3 style="color: #2196F3;">Naive Bayes Prediksi Obat: <strong>{drug_encoder.inverse_transform([prediction_gnb])[0]}</strong></h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Evaluation Metrics
st.markdown("### 📊 **Evaluasi Model**")
st.write("**KNN Accuracy**: {:.2f}%".format(accuracy_score(y_test, knn.predict(X_test)) * 100))
st.write("**Naive Bayes Accuracy**: {:.2f}%".format(accuracy_score(y_test, gnb.predict(X_test)) * 100))

st.markdown("### 📋 **Classification Report (KNN)**")
st.text(classification_report(y_test, knn.predict(X_test)))

st.markdown("### 📋 **Classification Report (Naive Bayes)**")
st.text(classification_report(y_test, gnb.predict(X_test)))

st.sidebar.info("Dibuat oleh Naufal Ramadhan - 2025")
