import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Configurar la página
st.set_page_config(page_title="Clasificación de Enfermedades en Hojas", page_icon="🌿", layout="centered")

st.title("🌱 Clasificación de Enfermedades en Hojas")
st.write("Sube una imagen de una hoja afectada o usa la cámara para capturarla. El modelo te dirá la enfermedad detectada.")

# URL del modelo en Google Drive (con nuevo ID)
modelo_url = "https://drive.google.com/uc?id=1ydcoy8oSuKDu_zL3kloj6fjHbZNHQtSr"
modelo_path = "modelo_vgg16_citrus.h5"

# Verificar si el modelo ya está descargado
if not os.path.exists(modelo_path):
    st.info("Descargando el modelo, por favor espera...")
    try:
        gdown.download(modelo_url, modelo_path, quiet=False)
    except Exception as e:
        st.error(f"🚨 Error al descargar el modelo: {str(e)}")
        st.stop()

# Verificar si el archivo realmente existe después de la descarga
if os.path.exists(modelo_path):
    st.success("✅ Modelo descargado correctamente.")
else:
    st.error("🚨 El archivo del modelo no se encuentra después de la descarga.")
    st.stop()

# Cargar el modelo
try:
    modelo = tf.keras.models.load_model(modelo_path)
    st.success("✅ Modelo cargado exitosamente.")
except Exception as e:
    st.error(f"🚨 Error al cargar el modelo: {str(e)}")
    st.stop()

# Diccionario de clases en español
clases = {
    0: "Mancha negra (Black Spot)",
    1: "Cancro (Canker)",
    2: "Verdeamiento (Greening)",
    3: "Sano (Healthy)",
    4: "Melanosis (Melanose)"
}

