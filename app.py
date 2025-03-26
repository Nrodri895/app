import os
import gdown
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image

# 📌 URL del modelo en Google Drive (Reemplaza con tu ID correcto)
URL_DRIVE = "https://drive.google.com/uc?id=1mlL4yG-9pZWhTQi91ht7YB3sWGXc79Cr"
NOMBRE_MODELO = "modelo_vgg16_citrus.h5"

# 📌 Función para descargar el modelo si no está en el directorio
def descargar_modelo():
    if not os.path.exists(NOMBRE_MODELO):
        st.info("Descargando modelo desde Google Drive...")
        gdown.download(URL_DRIVE, NOMBRE_MODELO, quiet=False)
    else:
        st.success("Modelo ya descargado.")

# 📌 Cargar el modelo con caché en Streamlit
@st.cache_resource
def load_model():
    descargar_modelo()
    return tf.keras.models.load_model(NOMBRE_MODELO)

# 📌 Inicializar modelo
modelo = load_model()

# 📌 Función de predicción
def predecir_imagen(imagen):
    imagen = imagen.resize((224, 224))  # Ajustar tamaño a VGG16
    imagen_array = np.array(imagen) / 255.0  # Normalizar
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir batch dimension

    prediccion = modelo.predict(imagen_array)
    clases = ["Mancha negra", "Cancro", "Verdeamiento", "Sano", "Melanosis"]
    indice_predicho = np.argmax(prediccion)
    return clases[indice_predicho], prediccion[0][indice_predicho]

# 📌 Interfaz en Streamlit
st.title("🟢 Clasificación de Enfermedades en Hojas de Cítricos")

subida = st.file_uploader("📤 Sube una imagen de una hoja", type=["jpg", "png", "jpeg"])

if subida:
    imagen_pil = Image.open(subida)
    st.image(imagen_pil, caption="Imagen subida", use_column_width=True)

    # 📌 Hacer predicción
    resultado, confianza = predecir_imagen(imagen_pil)

    # 📌 Mostrar resultado
    st.subheader(f"🔍 Predicción: {resultado}")
    st.write(f"📊 Confianza: {confianza:.2%}")

