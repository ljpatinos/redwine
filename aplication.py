import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv("redwine.csv")  # Ajusta la ruta si es necesario
    return df

# Cargar modelo
def load_model():
    filename = "best_model.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Interfaz en Streamlit
def main():
    st.markdown(
        '<h1 style="color: #FFFFFF; text-align: center;">Predicción de la calidad del vino rojo </h1>',
        unsafe_allow_html=True
    )

    # Cargar datos
    df = load_data()

    # Mostrar estadísticas descriptivas
    st.subheader("📊 Estadísticas Descriptivas")
    st.write(df.describe())

    # Mostrar tipos de variables
    st.subheader("📌 Tipo de Variables")
    st.write(df.dtypes)

    # Gráfico de barras de la variable "quality"
    st.subheader("📈 Distribución de la Calidad del Vino")
    fig, ax = plt.subplots()
    sns.countplot(x=df["quality"], ax=ax, palette="viridis")
    ax.set_title("Distribución de la Calidad del Vino")
    ax.set_xlabel("Calidad")
    ax.set_ylabel("Frecuencia")
    st.pyplot(fig)

    # Inputs para predicción
    st.markdown("Ingrese las características para predecir la calidad:")

    fixed_acidity = st.number_input("Acidez fija", min_value=0.0, format="%.5f")
    volatile_acidity = st.number_input("Ácidez volátil", min_value=0.0, format="%.2f")
    citric_acid = st.number_input("Ácido cítrico", min_value=0.0, format="%.2f")
    residual_sugar = st.number_input("Azúcar residual", min_value=0.0, format="%.2f")
    chlorides = st.number_input("Cloruros", min_value=0.0, format="%.3f")
    free_sulfur_dioxide = st.number_input("Dióxido de azufre libre", min_value=1.0, format="%.2f")
    total_sulfur_dioxide = st.number_input("Dióxido de azufre total", min_value=0.0, format="%.1f")
    density = st.number_input("Densidad", min_value=0.0, format="%.3f")
    pH = st.number_input("pH", min_value=0.0, format="%.1f")
    sulphates = st.number_input("Sulfatos", min_value=0.0, format="%.1f")
    alcohol = st.number_input("Contenido de alcohol (%)", min_value=0.0, format="%.2f")

    clases = {0: 'bueno', 1: '', 2: 'nada', 3: '', 4: ''}

    # Botón de predicción
    if st.button("Predecir Calidad"):
        model = load_model()
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
        prediction = np.argmax(model.predict(input_data))
        st.markdown(f"### La calidad estimada del vino es: **{clases[prediction]}**")

if __name__ == "__main__":
    main()
