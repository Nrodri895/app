import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# Configurar página
st.set_page_config(page_title="Clasificación de Enfermedades en Hojas", page_icon="🌿", layout="centered")

# Títulos y descripción
st.title("🌱 Clasificación de Enfermedades en Hojas")
st.write("Sube una imagen de una hoja afectada o usa la cámara para capturarla. El modelo te dirá la enfermedad detectada.")

def load_model():
    url = "https://drive.google.com/uc?id=1A2B3C4D5E6F7G8H"
    output = "modelo_vgg16_citrus.h5"

    # Verificar si el archivo ya existe
    if not os.path.exists(output):
        print("Descargando modelo...")
        gdown.download(url, output, quiet=False)

    # Mostrar los archivos en el directorio actual
    print("Archivos en el directorio:", os.listdir())

    # Verificar si la descarga fue exitosa
    if os.path.exists(output):
        print(f"Archivo encontrado: {output}")
        return tf.keras.models.load_model(output)
    else:
        raise FileNotFoundError(f"No se encontró el archivo {output}")

modelo = load_model()

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

    # Verificar forma de imagen antes de pasarla al modelo
    st.write(f"Forma de la imagen de entrada: {imagen_array.shape}")

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

