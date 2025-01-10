# import cv2

# def verificar_acceso_webcam(camara_id=0):
#     # Intentar abrir la cámara
#     captura = cv2.VideoCapture(camara_id)
    
#     if not captura.isOpened():
#         print(f"No se pudo acceder a la cámara con ID {camara_id}.")
#         return False
    
#     print(f"Cámara con ID {camara_id} accesible.")
    
#     # Mostrar la vista previa de la cámara
#     while True:
#         ret, frame = captura.read()
#         if not ret:
#             print("No se pudo capturar un cuadro de la cámara.")
#             break
        
#         # Mostrar el cuadro en una ventana
#         cv2.imshow("Vista previa de la cámara", frame)
        
#         # Presiona 'q' para salir
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Saliendo...")
#             break
    
#     # Liberar recursos
#     captura.release()
#     cv2.destroyAllWindows()
#     return True

# if __name__ == "__main__":
#     verificar_acceso_webcam(2)


import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo acceder a la cámara.")
else:
    print("Cámara accesible.")
cap.release()
cv2.destroyAllWindows()
