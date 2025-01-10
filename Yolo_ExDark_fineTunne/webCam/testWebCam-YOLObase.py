import cv2
from ultralytics import YOLO

def yolo_en_webcam(modelo_path, camara_id=0):
    # Cargar el modelo YOLO
    model = YOLO(modelo_path)

    # Abrir la cámara
    cap = cv2.VideoCapture(camara_id)
    if not cap.isOpened():
        print(f"No se pudo acceder a la cámara con ID {camara_id}.")
        return

    print("Mostrando detecciones en tiempo real. Presiona 'q' para salir.")
    while cap.isOpened():
        # Capturar frame de la cámara
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar el frame de la cámara.")
            break

        # Redimensionar el frame (opcional)
        resized_frame = cv2.resize(frame, (640, 640))

        # Realizar predicción con YOLO
        results = model(resized_frame)

        # Dibujar las detecciones en el frame
        annotated_frame = results[0].plot()

        # Mostrar el frame anotado
        cv2.imshow('Detecciones YOLO', annotated_frame)

        # Salir si se presiona la tecla 'q'
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            print("Cerrando la transmisión...")
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    modelo_path = "best.pt"  # Ruta al modelo entrenado
    yolo_en_webcam(modelo_path)
