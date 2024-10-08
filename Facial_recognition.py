import cv2
import face_recognition
import os
import pandas as pd

# Ruta de la carpeta que contiene las imágenes
ruta_carpeta = "C:/Users/diana/OneDrive/Escritorio/FOTOS/"

# Verificar que la ruta existe
if not os.path.exists(ruta_carpeta):
    raise Exception(f"La ruta {ruta_carpeta} no existe.")

codificaciones = {}
for archivo in os.listdir(ruta_carpeta):
    if archivo.lower().endswith((".jpg", ".png", ".jpeg")):
        ruta_imagen = os.path.join(ruta_carpeta, archivo)
        imagen = face_recognition.load_image_file(ruta_imagen)
        encodings = face_recognition.face_encodings(imagen)
        if encodings:
            codificaciones[archivo] = encodings[0]
        else:
            print(f"No se encontraron caras en {archivo}.")

# Lista para almacenar los resultados
resultados = []

# Iniciar la cámara
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise Exception("No se pudo acceder a la cámara.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break
    frame = cv2.flip(frame, 1)

    # Obtener ubicaciones de las caras
    face_locations = face_recognition.face_locations(frame, model="hog")

    # Procesar cada cara detectada
    for face_location in face_locations:
        encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])
        if encodings:
            face_frame_encodings = encodings[0]
            nombre_archivo = "Desconocido"

            # Comparar con cada imagen de referencia
            for archivo, codificacion in codificaciones.items():
                result = face_recognition.compare_faces([codificacion], face_frame_encodings, tolerance=0.6)
                if result[0]:
                    nombre_archivo = archivo
                    break

            # Dibujar rectángulo y nombre en el frame
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, nombre_archivo, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

            # Imprimir o almacenar el nombre del archivo
            print("Coincidencia:", nombre_archivo)
            resultados.append({'Nombre_Archivo': nombre_archivo})
        else:
            print("No se pudo obtener la codificación de la cara detectada.")

    # Mostrar el frame procesado
    cv2.imshow('Video', frame)

    # Terminar la captura con la tecla 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

# Convertir los resultados en un DataFrame
resultados_df = pd.DataFrame(resultados)
