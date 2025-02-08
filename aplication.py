import streamlit as st
import numpy as np
import pickle
import gzip

def load_model():
    filename = "best_model.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    # Título con color
    st.markdown(
        '<h1 style="color: #FFFFFF; text-align: center;">Predicción de la calidad del vino rojo </h1>',
        unsafe_allow_html=True
    )

    st.markdown("El mejor modelo de regresión obtenido presentó las siguientes características KernelRidge, Scaler: StandardScaler, alpha=0.1, kernel='rbf' y MAE=0.7414")
    
    st.markdown("Ingrese las características de la casa para predecir la calidad:")
    
    # Entradas para las 13 características del conjunto de datos Boston Housing
    fixed_acidity = st.number_input("Acidez fija", min_value=0.0, format="%.5f")
    volatile_acidity = st.number_input("Ácidez volatil", min_value=0.0, format="%.2f")
    citric_acid = st.number_input("Ácido cítrico", min_value=0.0, format="%.2f")
    residual_sugar = st.number_input("Azucar residual", min_value=0.0, format="%.2f")
    chlorides = st.number_input("Cloruros", min_value=0.0, format="%.3f")
    free_sulfur_dioxide = st.number_input("Dioxido de azufre libre", min_value=1.0, format="%.2f")
    total_sulfur_dioxide = st.number_input("Dioxido de azufre total", min_value=0.0, format="%.1f")
    density = st.number_input("Densidad", min_value=0.0, format="%.3f")
    pH = st.number_input("pH", min_value=0.0, format="%.1f")
    sulphates = st.number_input("Sulfatos", min_value=0.0, format="%.1f")
    alcohol = st.number_input("Contenido de alcohol (%)", min_value=0.0, format="%.2f")
        
    # Botón de predicción
    if st.button("Predecir Precio"):
        model = load_model()
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
        prediction = model.predict(input_data)
        st.markdown(f"### Precio estimado de la vivienda: **${prediction}**")

if __name__ == "__main__":
    main()
