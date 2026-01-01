import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
from gesture_detector import GestureDetector


def main():
    # Inicializar detector con detección de emociones habilitada
    detector = GestureDetector(
        draw_face=True,
        draw_pose=True,
        draw_hands=True,
        enable_emotion_detection=True
    )

    print("Sistema de Detección de Gestos y Emociones Iniciado")
    print("Gestos cargados:", len(detector.list_gestures()))
    print("Presiona 'q' para salir")
    print("Presiona 'g' para ver gestos disponibles")
    print("Presiona 's' para ver estadísticas")
    print("Presiona 'r' para resetear detector")
    print("Presiona '1' para alternar landmarks de cara")
    print("Presiona '2' para alternar landmarks de cuerpo")
    print("Presiona '3' para alternar landmarks de manos")
    print("Presiona 'e' para activar/desactivar detección de emociones")

    # ----------------------------
    # Inicializar cámara
    # ----------------------------
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: No se pudo abrir ninguna cámara")
        return

    print("✅ Cámara abierta exitosamente")

    # ----------------------------
    # Crear ventana UNA SOLA VEZ
    # ----------------------------
    window_name = 'Detección de Gestos y Emociones - Presiona Q para salir'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    # ----------------------------
    # Bucle principal
    # ----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: No se pudo leer el frame")
            break

        frame = cv2.flip(frame, 1)

        # Procesar frame
        processed_frame, data = detector.process_frame(frame)

        cv2.imshow(window_name, processed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('g'):
            print("\nGESTOS DISPONIBLES:")
            for i, gesture in enumerate(detector.list_gestures()):
                print(f"  {i + 1}. {gesture}")

        elif key == ord('s'):
            stats = detector.get_statistics()
            print("\nESTADÍSTICAS:")
            if not stats:
                print("  Aún no hay suficientes datos")
            for k, v in stats.items():
                print(f"  {k}: {v}")

        elif key == ord('r'):
            if detector.emotion_detector:
                detector.emotion_detector.reset()
            print("Detector resetado")

        elif key == ord('1'):
            detector.draw_face = not detector.draw_face
            print(f"Landmarks → Cara={detector.draw_face}, "
                  f"Cuerpo={detector.draw_pose}, Manos={detector.draw_hands}")

        elif key == ord('2'):
            detector.draw_pose = not detector.draw_pose
            print(f"Landmarks → Cara={detector.draw_face}, "
                  f"Cuerpo={detector.draw_pose}, Manos={detector.draw_hands}")

        elif key == ord('3'):
            detector.draw_hands = not detector.draw_hands
            print(f"Landmarks → Cara={detector.draw_face}, "
                  f"Cuerpo={detector.draw_pose}, Manos={detector.draw_hands}")

        elif key == ord('e'):
            detector.enable_emotion_detection = not detector.enable_emotion_detection
            status = "ON" if detector.enable_emotion_detection else "OFF"
            print(f"Detección de emociones: {status}")

    # ----------------------------
    # Liberar recursos
    # ----------------------------
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nSistema cerrado exitosamente")


if __name__ == "__main__":
    main()
