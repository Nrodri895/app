import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import h5py

# Configurar página
st.set_page_config(page_title="Clasificación de Enfermedades en Hojas", page_icon="🌿", layout="centered")

# Títulos y descripción
st.title("🌱 Clasificación de Enfermedades en Hojas")
st.write("Sube una imagen de una hoja afectada o usa la cámara para capturarla. El modelo te dirá la enfermedad detectada.")

# Ruta del modelo
modelo_path = "modelo_vgg16_citrus.h5"

# URL del modelo en Google Drive (reemplaza con el ID correcto)
modelo_url = "https://drive.google.com/file/d/1mlL4yG-9pZWhTQi91ht7YB3sWGXc79Cr/view?usp=sharing"

# Verificar si el modelo existe, si no, descargarlo
if not os.path.exists(modelo_path):
    st.warning("🔄 Descargando el modelo... Esto puede tardar unos minutos.")
    gdown.download(modelo_url, modelo_path, quiet=False)

# Intentar cargar el modelo
try:
    modelo = tf.keras.models.load_model(modelo_path)
    st.success("✅ Modelo cargado correctamente.")
except Exception as e:
    st.error(f"🚨 Error al cargar el modelo: {e}")
    st.stop()

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

    return clases.get(clase_predicha, "Clase desconocida")  # Retornar el nombre en español

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

# Estilo mejorado
st.markdown(
    """
    <style>
        .stApp { background-color: #e8f5e9; }
        .stButton>button { background-color: #4caf50; color: white; font-size: 18px; border-radius: 10px; }
        .stRadio>div { display: flex; justify-content: center; }
    </style>
    """,
    unsafe_allow_html=True
)

