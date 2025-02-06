import streamlit as st
import numpy as np
import pickle
import gzip

def load_model():
    filename = "model_trained_regressor.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

def main():
    # Título con color
    st.markdown(
        '<h1 style="color: #4CAF50; text-align: center;">Predicción de Precios de Casas (Boston Housing)</h1>',
        unsafe_allow_html=True
    )

    st.markdown("El mejor modelo de regresión obtenido presentalas siguientes características KernelRidge, Scaler: StandardScaler, alpha=0.1, kernel='rbf' y MAE=0.7414")
    
    st.markdown("Ingrese las características de la casa para predecir su precio:")
    
    # Entradas para las 13 características del conjunto de datos Boston Housing
    crim = st.number_input("Tasa de criminalidad per cápita por ciudad", min_value=0.0, format="%.5f")
    zn = st.number_input("Proporción de terreno residencial zonificado para lotes grandes", min_value=0.0, format="%.2f")
    indus = st.number_input("Proporción de acres comerciales por ciudad", min_value=0.0, format="%.2f")
    chas = st.selectbox("Variable ficticia Charles River (1: colinda, 0: no colinda)", [0, 1])
    nox = st.number_input("Concentración de óxidos de nitrógeno (partes por 10 millones)", min_value=0.0, format="%.3f")
    rm = st.number_input("Número promedio de habitaciones por vivienda", min_value=1.0, format="%.2f")
    age = st.number_input("Proporción de viviendas construidas antes de 1940", min_value=0.0, format="%.1f")
    dis = st.number_input("Distancia ponderada a cinco centros de empleo de Boston", min_value=0.0, format="%.3f")
    rad = st.number_input("Índice de accesibilidad a carreteras radiales", min_value=1, max_value=24, step=1)
    tax = st.number_input("Tasa de impuesto a la propiedad por cada $10,000", min_value=0.0, format="%.1f")
    ptratio = st.number_input("Índice alumno-profesor por ciudad", min_value=0.0, format="%.2f")
    b = st.number_input("Proporción de población afroamericana por ciudad", min_value=0.0, format="%.2f")
    lstat = st.number_input("Porcentaje de población de bajos ingresos", min_value=0.0, format="%.2f")
    
    # Botón de predicción
    if st.button("Predecir Precio"):
        model = load_model()
        input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat]])
        prediction = model.predict(input_data)
        st.markdown(f"### Precio estimado de la vivienda: **${prediction}**")

if __name__ == "__main__":
    main()
