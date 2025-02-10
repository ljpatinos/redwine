import streamlit as st
import numpy as np
import pandas as pd
import pickle
import gzip
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de estilo
st.set_page_config(page_title="Predicción de Calidad del Vino", layout="wide")
st.markdown(
    """
    <style>
        body { background-color: #4F4F99; }
        h1, h2, h3, h4, h5, h6 { color: #FFFFFF; }
        .stSidebar { background-color: #561B47; }
        .dataframe { margin: auto; } /* Centrar la tabla */
    </style>
    """, unsafe_allow_html=True
)

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

    # Barra lateral para seleccionar variable
    st.sidebar.header("🔍 Exploración de Variables")
    selected_var = st.sidebar.selectbox("Selecciona una variable:", df.columns)

    # Mostrar estadísticas descriptivas
    st.sidebar.subheader(f"📊 Estadísticas Descriptivas de '{selected_var}'")
    st.sidebar.write(df[selected_var].describe())
    
    st.sidebar.subheader("📌 Tipo de Variable")
    st.sidebar.write(f"La variable '{selected_var}' es de tipo: **{df[selected_var].dtype}**")

        # Generar gráficos
    st.subheader("📈 Visualización de la Variable")

    # Boxplot
    st.markdown("### 🔲 Boxplot")
    fig, ax = plt.subplots()
    sns.boxplot(y=df[selected_var], ax=ax, color="lightblue")
    ax.set_title(f"Boxplot de {selected_var}")
    st.pyplot(fig)

    # Gráfico de barras (solo si la variable es categórica o tiene pocos valores únicos)
    if df[selected_var].nunique() < 10:
        st.markdown("### 📊 Gráfico de Barras")
        fig, ax = plt.subplots()
        sns.countplot(x=df[selected_var], ax=ax, palette="viridis")
        ax.set_title(f"Distribución de {selected_var}")
        st.pyplot(fig)

    # Dispersión contra calidad (si es numérica)
    if df[selected_var].dtype in ["int64", "float64"] and selected_var != "quality":
        st.markdown("### 🔵 Gráfico de Dispersión vs Calidad")
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[selected_var], y=df["quality"], ax=ax, alpha=0.5)
        ax.set_title(f"Relación entre {selected_var} y Calidad")
        st.pyplot(fig)

    # Sección de predicción de calidad
    st.markdown("---")
    st.subheader("🎯 Predicción de Calidad del Vino")
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
