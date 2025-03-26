import os
import gdown
import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image


URL_DRIVE = "https://drive.google.com/uc?id=1ydcoy8oSuKDu_zL3kloj6fjHbZNHQtSr"
NOMBRE_MODELO = "modelo_vgg16_citrus.h5"


def descargar_modelo():
    if not os.path.exists(NOMBRE_MODELO):
        st.info("Descargando modelo desde Google Drive...")
        gdown.download(URL_DRIVE, NOMBRE_MODELO, quiet=False)
    else:
        st.success("Modelo ya descargado.")

#  Cargar el modelo con caché en Streamlit
@st.cache_resource
def load_model():
    descargar_modelo()
    return tf.keras.models.load_model(NOMBRE_MODELO)

#  Inicializar modelo
modelo = load_model()

# Función de predicción
def predecir_imagen(imagen):
    imagen = imagen.resize((224, 224))  # Ajustar tamaño a VGG16
    imagen_array = np.array(imagen) / 255.0  # Normalizar
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir batch dimension

    prediccion = modelo.predict(imagen_array)
    clases = ["Mancha negra", "Cancro", "Verdeamiento", "Sano", "Melanosis"]
    indice_predicho = np.argmax(prediccion)
    return clases[indice_predicho], prediccion[0][indice_predicho]

#  Interfaz en Streamlit
st.title("🟢 Clasificación de Enfermedades en Hojas de Cítricos")

# 📷 Opción para subir imagen o usar la cámara
opcion = st.radio(" Selecciona una opción:", ["Subir imagen", "Usar cámara"])

imagen_pil = None  # Variable para almacenar la imagen

if opcion == "Subir imagen":
    subida = st.file_uploader("📤 Sube una imagen de una hoja", type=["jpg", "png", "jpeg"])
    if subida:
        imagen_pil = Image.open(subida)

elif opcion == "Usar cámara":
    captura = st.camera_input("📸 Toma una foto")
    if captura:
        imagen_pil = Image.open(captura)

#  Si hay imagen, hacer predicción
if imagen_pil:
    st.image(imagen_pil, caption="📷 Imagen seleccionada", use_column_width=True)

    #  Hacer predicción
    resultado, confianza = predecir_imagen(imagen_pil)

    #  Mostrar resultado
    st.subheader(f"🔍 Predicción: {resultado}")
    st.write(f"📊 Confianza: {confianza:.2%}")

