!pip install streamlit joblib scikit-learn pandas

'''
se ejecuta primero cargan las bibliotecas requeridas con
pip install -r requirements.txt (tenga creado ese archivo)

se ejecuta la aplicacion con
steamlit run app.py desde la linea del terminal o de su pwershell
'''

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Descargar los archivos del modelo y el scaler (simulado en Colab)
# En un entorno local, estos archivos ya estarían en tu directorio.
# from google.colab import files
# uploaded = files.upload()
# for fn in uploaded.keys():
#   print('User uploaded file "{name}"'.format(name=fn))


# Cargar el modelo y el scaler
try:
    svc_model = joblib.load('svc_model.jb')
    scaler = joblib.load('scaler.jb')
except FileNotFoundError:
    st.error("Error: Los archivos del modelo (svc_model.jb y scaler.jb) no se encontraron.")
    st.stop() # Detiene la ejecución de la aplicación si los archivos no se encuentran

# Título e imagen de banner
st.image("fibrilacao-atrial-imagem-destacada-733x412.jpg", use_column_width=True)
st.title("Modelo IA para predicción de problemas cardiacos")

# Resumen del funcionamiento del modelo
st.write("""
Esta aplicación utiliza un modelo de Inteligencia Artificial basado en Máquinas de Vectores de Soporte (SVC)
para predecir la probabilidad de que un paciente sufra problemas cardíacos.
El modelo fue entrenado utilizando datos de edad y colesterol, que fueron escalados previamente
para mejorar el rendimiento del modelo.
""")

# Sidebar para la entrada del usuario
st.sidebar.header("Datos del Paciente")

edad = st.sidebar.slider("Edad (años)", min_value=20, max_value=100, value=20, step=1)
colesterol = st.sidebar.slider("Colesterol (mg/dL)", min_value=120, max_value=600, value=200, step=10)

# Crear un DataFrame con los datos de entrada
input_data = pd.DataFrame({'edad': [edad], 'colesterol': [colesterol]})

# Escalar los datos de entrada usando el scaler cargado
input_data_scaled = scaler.transform(input_data)
input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=['edad', 'colesterol'])


# Realizar la predicción
prediction = svc_model.predict(input_data_scaled_df)

# Mostrar los resultados
st.subheader("Resultado de la Predicción")

if prediction[0] == 0:
    st.markdown(
        """
        <div style='background-color:#d4edda; color:#155724; padding:10px; border-radius:5px;'>
            <p style='font-size:20px; font-weight:bold;'>Resultado: No sufrirá del corazón 😊</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image("cuando-debo-visitar-al-cardiologo-1.jpg", caption="¿Cuándo visitar al cardiólogo?")
else:
    st.markdown(
        """
        <div style='background-color:#cfe2ff; color:#084298; padding:10px; border-radius:5px;'>
            <p style='font-size:20px; font-weight:bold;'>Resultado: Sufrirá del corazón 😞</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.image("insuficiencia-cardiaca.jpg", caption="Insuficiencia Cardíaca")

# Información del autor
st.markdown("---")
st.write("Elaborado por: Alfredo Diaz © unab 2025")