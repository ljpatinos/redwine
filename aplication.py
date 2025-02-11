import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request

import streamlit as st

st.set_page_config(page_title="Predicción de Calidad del Vino", layout="wide")

st.markdown(
    """
    <style>
        body { background-color: #4F4F99; }
        .titulo {
            color: #FFFFFF; 
            font-size: 35px;  
            font-weight: bold;
            text-align: center;
            margin-bottom: -10px; /* Reduce el espacio debajo del título */
        }
        h2 { color: #FFFFFF; font-size: 24px; }
        h3, h4, h5, h6 { color: #FFFFFF; font-size: 18px; }
        .stSidebar { background-color: #561B47; }
        .dataframe { margin: auto; }
    </style>
    """, 
    unsafe_allow_html=True
)

# Usa Markdown sin el anclaje automático de Streamlit
st.markdown('<div class="titulo">Predicción de la calidad del vino rojo</div>', unsafe_allow_html=True)

# Cargar los datos
@st.cache_data
def load_data():
    df = pd.read_csv("redwine.csv")  # Ajusta la ruta si es necesario
    return df

MODEL_URLS = {
    "Red neuronal": "https://raw.githubusercontent.com/ljpatinos/redwine/main/best_model.pkl.gz",
    "Arbol de decisiones": "https://raw.githubusercontent.com/ljpatinos/redwine/main/model_trained_DT.pkl.gz",
    "Arbol (6Var)": "https://raw.githubusercontent.com/usuario/ljpatinos/redwine/main/svm.pkl.gz"
}
def load_model(url):
    with urllib.request.urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as f:
            model = pickle.load(f)
    return model

# Interfaz en Streamlit
def main():

    # Cargar datos
    df = load_data()

    # Barra lateral para seleccionar variable
    st.sidebar.header("🔍 Exploración de Variables", divider='gray')
    selected_var = st.sidebar.selectbox("Selecciona una variable:", df.columns)

    # Mostrar estadísticas descriptivas
    st.sidebar.subheader(f"📊 Estadísticas Descriptivas de '{selected_var}'", divider='gray')
    st.sidebar.write(df[selected_var].describe())
    
    # Selección del modelo en la barra lateral
    st.sidebar.header("🔍 Seleccionar Modelo de Predicción")
    selected_model_name = st.sidebar.selectbox("Elige un modelo:", list(MODEL_URLS.keys()))
    selected_model_url = MODEL_URLS[selected_model_name]

    # Cargar el modelo seleccionado
    model = load_model(selected_model_url)

    # Generar gráficos
    # Histograma 
    st.markdown("---")
    st.subheader("📊 Histograma", divider='gray')
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(df[selected_var], bins=20, kde=True, color="blue", ax=ax)
    ax.set_title(f"Histograma de {selected_var}", fontsize=6)
    st.pyplot(fig)

        # Título
    # Sección de predicción de calidad
    st.markdown("---")
    st.subheader("📈 Dispersión", divider='gray')
    
    # Colores para la calidad
    palette = sns.color_palette("coolwarm", as_cmap=False)  # Mapa de colores

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = sns.scatterplot(
        data=df,
        x="quality",
        y=df[selected_var],
        hue="quality",
        palette=palette,
        alpha=0.7
    )

    # Personalizar
    plt.xlabel("Quality")
    plt.ylabel(f"{selected_var}")
    plt.title(f"Relación entre {selected_var} y Calidad del Vino")
    plt.legend(title="Calidad", bbox_to_anchor=(1, 1))
    # Mostrar en Streamlit
    st.pyplot(fig)
    
    st.markdown("---")
    image_url = "https://raw.githubusercontent.com/ljpatinos/redwine/main/correlation_matrix.png"  # 🔹 Reemplaza con tu URL correcta
    st.subheader("🔢Correlación", divider='gray')
    st.image(image_url, caption="Matriz de Correlación")

    # Sección de predicción de calidad
    st.markdown("---")
    st.subheader("🎯 Predicción", divider='gray')
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
    #if st.button("Predecir Calidad"):
        #model = load_model()
        #input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                                #free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
        #prediction = np.argmax(model.predict(input_data))
        #st.markdown(f"### La calidad estimada del vino es: **{clases[prediction]}**")

    # Resultados con otros métodos
    st.markdown("---")
    st.subheader("🧮 Histograma", divider='gray')
    
    if st.button("Predecir Calidad"):
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                            free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]])
        prediction = np.argmax(model.predict(input_data))
        st.markdown(f"### La calidad estimada del vino con **{selected_model_name}** es: **{clases[prediction]}**")

if __name__ == "__main__":
    main()
