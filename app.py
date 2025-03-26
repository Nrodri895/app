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

# URL del modelo en Google Drive
modelo_url = "https://drive.google.com/uc?id=1mlL4yG-9pZWhTQi91ht7YB3sWGXc79Cr"
modelo_path = "modelo_vgg16_citrus.h5"

# Verificar si el modelo ya está descargado
if not os.path.exists(modelo_path):
    st.info("Descargando el modelo, por favor espera...")
    gdown.download(modelo_url, modelo_path, quiet=False)

# Cargar el modelo
try:
    modelo = tf.keras.models.load_model(modelo_path)
    st.success("✅ Modelo cargado exitosamente.")
except Exception as e:
    st.error(f"🚨 Error al cargar el modelo: {str(e)}")

# Diccionario de clases en español
clases = {
    0: "Mancha negra (Black Spot)",
    1: "Cancro (Canker)",
    2: "Verdeamiento (Greening)",
    3: "Sano (Healthy)",
    4: "Melanosis (Melanose)"
}

def predecir_imagen(imagen_pil):
    imagen_pil = imagen_pil.resize((224, 224))  # Redimensionar a 224x224
    imagen_array = np.array(imagen_pil) / 255.0  # Normalizar

    # Asegurar que la imagen tenga 3 canales (RGB)
    if imagen_array.shape[-1] != 3:
        imagen_array = np.stack((imagen_array,)*3, axis=-1)

    # Agregar dimensión batch (1, 224, 224, 3)
    imagen_array = np.expand_dims(imagen_array, axis=0)

    # Realizar predicción
    prediccion = modelo.predict(imagen_array)
    clase_predicha = np.argmax(prediccion)  # Obtener la clase con mayor probabilidad

    return clases[clase_predicha]  # Retornar el nombre en español

# Subida de imagen o captura con cámara
opcion = st.radio("Selecciona una opción:", ("Subir una imagen", "Tomar foto con la cámara"))

if opcion == "Subir una imagen":
    imagen_subida = st.file_uploader("Carga una imagen", type=["jpg", "jpeg", "png"])
    if imagen_subida is not None:
        imagen_pil = Image.open(imagen_subida)
        st.image(imagen_pil, caption="Imagen cargada", use_container_width=True)
        resultado = predecir_imagen(imagen_pil)
        st.success(f"🔍 **Resultado:** {resultado}")

elif opcion == "Tomar foto con la cámara":
    imagen_capturada = st.camera_input("Captura una imagen")
    if imagen_capturada is not None:
        imagen_pil = Image.open(imagen_capturada)
        st.image(imagen_pil, caption="Imagen capturada", use_container_width=True)
        resultado = predecir_imagen(imagen_pil)
        st.success(f"🔍 **Resultado:** {resultado}")
