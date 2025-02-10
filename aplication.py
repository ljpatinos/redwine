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
    df.rename(columns={
        "fixed_acidity": "Acidez Fija",
        "volatile_acidity": "Ácidez Volátil",
        "citric_acid": "Ácido Cítrico",
        "residual_sugar": "Azúcar Residual",
        "chlorides": "Cloruros",
        "free_sulfur_dioxide": "Dióxido de Azufre Libre",
        "total_sulfur_dioxide": "Dióxido de Azufre Total",
        "density": "Densidad",
        "pH": "pH",
        "sulphates": "Sulfatos",
        "alcohol": "Contenido de Alcohol",
        "quality": "Calidad"
    }, inplace=True)
    return df

# Cargar modelo
def load_model():
    filename = "best_model.pkl.gz"
    with gzip.open(filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Interfaz en Streamlit
def main():
    st.markdown('<h1 style="text-align: center;">Predicción de la Calidad del Vino Rojo</h1>', unsafe_allow_html=True)

    # Cargar datos
    df = load_data()

    # Barra lateral para selección
    st.sidebar.header("🔍 Exploración de Datos")
    selected_var = st.sidebar.selectbox("Selecciona una variable:", df.columns)
    selected_chart = st.sidebar.radio("Selecciona el tipo de gráfico:", ["Barras", "Boxplot", "Dispersión"])
    
    # Mostrar estadísticas descriptivas
    st.subheader(f"📊 Estadísticas Descriptivas de '{selected_var}'")
    st.write(df[[selected_var]].describe().style.set_properties(**{'text-align': 'center'}))

    # Mostrar tipo de variable
    st.subheader("📌 Tipo de Variable")
    st.write(f"La variable '{selected_var}' es de tipo: **{df[selected_var].dtype}**")

    # Generar gráficos
    st.subheader("📈 Visualización de la Variable")
    fig, ax = plt.subplots()
    
    if selected_chart == "Boxplot":
        sns.boxplot(y=df[selected_var], ax=ax, color="#ffcccb")
        ax.set_title(f"Boxplot de {selected_var}", color='white')
    elif selected_chart == "Barras" and df[selected_var].nunique() < 10:
        sns.countplot(x=df[selected_var], ax=ax, palette="viridis")
        ax.set_title(f"Distribución de {selected_var}", color='white')
    elif selected_chart == "Dispersión" and df[selected_var].dtype in ["int64", "float64"] and selected_var != "Calidad":
        sns.scatterplot(x=df[selected_var], y=df["Calidad"], ax=ax, alpha=0.5)
        ax.set_title(f"Relación entre {selected_var} y Calidad", color='white')
    else:
        st.write("El gráfico seleccionado no es aplicable a esta variable.")
    
    st.pyplot(fig)

    # Sección de predicción de calidad
    st.markdown("---")
    st.subheader("🎯 Predicción de Calidad del Vino")
    st.markdown("Ingrese las características para predecir la calidad:")

    # Entradas para predicción
    inputs = {}
    feature_names = ["Acidez Fija", "Ácidez Volátil", "Ácido Cítrico", "Azúcar Residual", "Cloruros", "Dióxido de Azufre Libre", "Dióxido de Azufre Total", "Densidad", "pH", "Sulfatos", "Contenido de Alcohol"]
    feature_keys = ["Acidez Fija", "Ácidez Volátil", "Ácido Cítrico", "Azúcar Residual", "Cloruros", "Dióxido de Azufre Libre", "Dióxido de Azufre Total", "Densidad", "pH", "Sulfatos", "Contenido de Alcohol"]
    
    for name, key in zip(feature_names, feature_keys):
        inputs[key] = st.number_input(name, min_value=0.0, format="%.3f")
    
    # Botón de predicción
    if st.button("Predecir Calidad"):
        model = load_model()
        input_data = np.array([[inputs[key] for key in feature_keys]])
        prediction = np.argmax(model.predict(input_data))
        calidad = {0: 'Baja', 1: 'Media', 2: 'Alta'}
        st.markdown(f"### La calidad estimada del vino es: **{calidad.get(prediction, 'Desconocida')}**")

if __name__ == "__main__":
    main()
